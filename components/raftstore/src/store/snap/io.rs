// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.
use std::{
    cell::RefCell,
    fs,
    io::{self, BufReader, Read, Write},
    iter::Iterator,
    path::Path,
    sync::Arc,
    thread::sleep_ms,
    usize,
};

use encryption::{DataKeyManager, DecrypterReader, EncrypterWriter, Iv};
use engine_rocks::util;
use engine_traits::{
    iter_option, CfName, Error as EngineError, ExternalSstFileInfo, FileMetadata, IterOptions,
    Iterable, Iterator as EngineIterator, KvEngine, Mutable, RefIterable, SstCompressionType,
    SstReader, SstWriter, SstWriterBuilder, WriteBatch, CF_DEFAULT, CF_WRITE,
};
use fail::fail_point;
use file_system::{File, IoBytesTracker, IoType, OpenOptions, WithIoType};
use kvproto::encryptionpb::EncryptionMethod;
use tikv_util::{
    box_try,
    codec::bytes::{BytesEncoder, CompactBytesFromFileDecoder},
    debug, error, info,
    time::{Instant, Limiter},
};

use super::{CfFile, Error, IO_LIMITER_CHUNK_SIZE};

// This defines the number of bytes scanned before trigger an I/O limiter check.
// It is used instead of checking the I/O limiter for each scan to reduce cpu
// overhead.
const SCAN_BYTES_PER_IO_LIMIT_CHECK: usize = 8 * 1024;

/// Used to check a procedure is stale or not.
pub trait StaleDetector {
    fn is_stale(&self) -> bool;
}

/// Statistics for tracking the process of building SST files.
#[derive(Clone, Copy, Default)]
pub struct BuildStatistics {
    /// The total number of keys processed during the build.
    pub key_count: usize,

    /// The total size (in bytes) of key-value pairs processed.
    /// This represents the combined size of keys and values before any
    /// compression.
    pub total_kv_size: usize,

    /// The total size (in bytes) of the generated SST files after compression.
    /// This reflects the on-disk size of the output files.
    pub total_sst_size: usize,

    /// The total size (in bytes) of the raw data in plain text format.
    /// This represents the uncompressed size of the CF_LOCK data.
    pub total_plain_size: usize,
}

/// Build a snapshot file for the given column family in plain format.
/// If there are no key-value pairs fetched, no files will be created at `path`,
/// otherwise the file will be created and synchronized.
pub fn build_plain_cf_file<E>(
    cf_file: &mut CfFile,
    key_mgr: Option<&Arc<DataKeyManager>>,
    snap: &E::Snapshot,
    start_key: &[u8],
    end_key: &[u8],
) -> Result<BuildStatistics, Error>
where
    E: KvEngine,
{
    let cf = cf_file.cf;
    let path = cf_file.path.join(cf_file.gen_tmp_file_name(0));
    let path = path.to_str().unwrap();
    let mut file = Some(box_try!(
        OpenOptions::new().write(true).create_new(true).open(path)
    ));
    let mut encrypted_file: Option<EncrypterWriter<File>> = None;
    let mut should_encrypt = false;

    if let Some(key_mgr) = key_mgr {
        let enc_info = box_try!(key_mgr.new_file(path));
        let mthd = enc_info.method;
        if mthd != EncryptionMethod::Plaintext {
            let writer = box_try!(EncrypterWriter::new(
                file.take().unwrap(),
                mthd,
                &enc_info.key,
                box_try!(Iv::from_slice(&enc_info.iv)),
            ));
            encrypted_file = Some(writer);
            should_encrypt = true;
        }
    }

    let mut writer = if !should_encrypt {
        file.as_mut().unwrap() as &mut dyn Write
    } else {
        encrypted_file.as_mut().unwrap() as &mut dyn Write
    };

    let mut stats = BuildStatistics::default();
    box_try!(snap.scan(cf, start_key, end_key, false, |key, value| {
        stats.key_count += 1;
        stats.total_kv_size += key.len() + value.len();
        box_try!(BytesEncoder::encode_compact_bytes(&mut writer, key));
        box_try!(BytesEncoder::encode_compact_bytes(&mut writer, value));
        Ok(true)
    }));

    if stats.key_count > 0 {
        cf_file.add_file(0);
        box_try!(BytesEncoder::encode_compact_bytes(&mut writer, b""));
        let file = if !should_encrypt {
            file.unwrap()
        } else {
            encrypted_file.unwrap().finalize().unwrap()
        };
        box_try!(file.sync_all());
        let metadata = box_try!(file.metadata());
        stats.total_plain_size += metadata.len() as usize;
    } else {
        drop(file);
        box_try!(fs::remove_file(path));
    }

    Ok(stats)
}
#[derive(Debug, PartialEq)]
pub struct SstFileInfo {
    pub file_name: String,
    pub smallest_key: Vec<u8>,
    pub largest_key: Vec<u8>,
    pub num_entries: u64,
}

fn get_lmax(files_metadata_by_level: &Vec<Vec<FileMetadata>>) -> u32 {
    for i in (0..files_metadata_by_level.len()).rev() {
        if files_metadata_by_level[i].len() > 0 {
            return i as u32;
        }
    }

    0
}

fn try_filter_lmax<E>(
    cf_file: &mut CfFile,
    engine: &E,
    start_key: &[u8],
    end_key: &[u8],
    file_id: &mut usize,
    files_metadata_by_level: Vec<Vec<FileMetadata>>,
) -> Result<(usize, Option<Vec<FileMetadata>>), Error>
where
    E: KvEngine,
{
    let cf = cf_file.cf;
    let lmax = get_lmax(&files_metadata_by_level) as usize;
    if lmax == 0 {
        return Ok((0, None));
    }

    let lmax_file_metadata = &files_metadata_by_level[lmax];

    // Filter files that belong to this region
    let region_files: Vec<_> = lmax_file_metadata
        .iter()
        .filter(|file| {
            let smallest = file.smallest_key.as_slice();
            let largest = file.largest_key.as_slice();
            smallest < end_key && largest > start_key
        })
        .cloned()
        .collect();
    let total_region_files = region_files.len(); // Total files belonging to the region

    // Find exclusive files
    let mut exclusive_files: Vec<FileMetadata> = region_files
        .into_iter()
        .filter(|file| {
            let smallest = file.smallest_key.as_slice();
            let largest = file.largest_key.as_slice();
            smallest >= start_key && largest <= end_key
        })
        .collect();

    let exclusive_count = exclusive_files.len(); // Count of exclusive files
    println!(
        "Region Lmax Info: total_region_files={}, exclusive_files={}",
        total_region_files, exclusive_count
    );

    // If the counts do not match, return early with lmax = 0 and an empty vector
    if total_region_files != exclusive_count {
        return Ok((0, None));
    }

    // Create hard links for each exclusive file
    for file in &mut exclusive_files {
        let src_path_str = format!("{}{}", engine.path(), file.name);
        let src_path = Path::new(&src_path_str);
        let dest_name = cf_file.gen_tmp_file_name(*file_id);
        let dest_path = cf_file.path.join(dest_name.clone());

        println!(
            "Hard link, source path: {:?}, dest path: {:?}",
            src_path, dest_path
        );

        fs::hard_link(&src_path, &dest_path).map_err(|e| {
            error!(
                "failed to create hard link for file in max level";
                "src" => src_path.display(),
                "dest" => dest_path.display(),
                "err" => ?e,
            );
            io::Error::new(io::ErrorKind::Other, format!("Hard link failed: {:?}", e))
        })?;

        file.name = dest_name;
        cf_file.add_file(*file_id); // Add the file to CfFile
        *file_id += 1;
    }

    Ok((lmax, Some(exclusive_files)))
}

fn scan_helper<Iter, F>(mut it: Iter, start_key: &[u8], mut f: F) -> Result<(), Error>
where
    Iter: engine_traits::Iterator,
    F: FnMut(&[u8], &[u8]) -> Result<bool, Error>,
{
    let mut remained = it.seek(start_key)?;
    while remained {
        remained = f(it.key(), it.value())? && it.next()?;
    }
    Ok(())
}

pub fn build_sst_cf_file_list_new<E>(
    cf_file: &mut CfFile,
    engine: &E,
    snap: &E::Snapshot,
    start_key: &[u8],
    end_key: &[u8],
    raw_size_per_file: u64,
    io_limiter: &Limiter,
    key_mgr: Option<Arc<DataKeyManager>>,
    for_balance: bool,
) -> Result<(BuildStatistics, Vec<FileMetadata>, Vec<SstFileInfo>), Error>
where
    E: KvEngine,
{
    let cf = cf_file.cf;
    let mut file_id: usize = 0;
    let mut iter_opt = iter_option(start_key, end_key, false);
    iter_opt.set_filter_lmax(true);
    let (mut iter, files_metadata_by_level) =
        snap.iterator_opt_and_get_metadata(cf, iter_opt).unwrap();
    // Handle lmax
    let (_, direct_ssts_info) = try_filter_lmax(
        cf_file,
        engine,
        start_key,
        end_key,
        &mut file_id,
        files_metadata_by_level,
    )?;

    let mut stats = BuildStatistics::default();
    if direct_ssts_info.is_none() {
        iter_opt = iter_option(start_key, end_key, false);
        iter = snap.iterator_opt(cf, iter_opt).unwrap();
    } else {
        cf_file.num_lmax_files = direct_ssts_info.clone().unwrap().len() as u32;
    }

    let mut generated_ssts_info = Vec::new();
    let mut remained_quota = 0;
    let mut path = cf_file
        .path
        .join(cf_file.gen_tmp_file_name(file_id))
        .to_str()
        .unwrap()
        .to_string();
    let sst_writer = RefCell::new(create_sst_file_writer::<E>(engine, cf, &path)?);
    let mut file_length: usize = 0;

    let finish_sst_writer = |sst_writer: E::SstWriter,
                             path: String,
                             key_mgr: Option<Arc<DataKeyManager>>|
     -> Result<SstFileInfo, Error> {
        let info = sst_writer.finish()?;
        (|| {
            fail_point!("inject_sst_file_corruption", |_| {
                static CALLED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if CALLED
                    .compare_exchange(
                        false,
                        true,
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                    )
                    .is_err()
                {
                    return;
                }
                // overwrite the file to break checksum
                let mut f = OpenOptions::new().write(true).open(&path).unwrap();
                f.write_all(b"x").unwrap();
            });
        })();

        let sst_reader = E::SstReader::open(&path, key_mgr)?;
        if let Err(e) = sst_reader.verify_checksum() {
            fs::remove_file(&path)?;
            error!(
                "failed to pass block checksum verification";
                "file" => path,
                "err" => ?e,
            );
            return Err(io::Error::new(io::ErrorKind::InvalidData, e).into());
        }
        File::open(&path).and_then(|f| f.sync_all())?;

        let sst_file_info = SstFileInfo {
            file_name: path.clone().rsplit('/').next().unwrap_or(&path).to_string(),
            smallest_key: info.smallest_key().to_vec(),
            largest_key: info.largest_key().to_vec(),
            num_entries: info.num_entries(),
        };
        Ok(sst_file_info)
    };

    let instant = Instant::now();
    let _io_type_guard = WithIoType::new(if for_balance {
        IoType::LoadBalance
    } else {
        IoType::Replication
    });

    let mut io_tracker = IoBytesTracker::new();
    let mut next_io_check_size = stats.total_kv_size + SCAN_BYTES_PER_IO_LIMIT_CHECK;
    let handle_read_io_usage = |io_tracker: &mut IoBytesTracker, remained_quota: &mut usize| {
        if let Some(io_bytes_delta) = io_tracker.update() {
            while io_bytes_delta.read as usize > *remained_quota {
                io_limiter.blocking_consume(IO_LIMITER_CHUNK_SIZE);
                *remained_quota += IO_LIMITER_CHUNK_SIZE;
            }
            *remained_quota -= io_bytes_delta.read as usize;
        }
    };

    let scan_fn = |key: &[u8], value: &[u8]| -> Result<bool, Error> {
        let entry_len = key.len() + value.len();
        if file_length + entry_len > raw_size_per_file as usize {
            cf_file.add_file(file_id); // add previous file
            file_length = 0;
            file_id += 1;
            let prev_path = path.clone();
            path = cf_file
                .path
                .join(cf_file.gen_tmp_file_name(file_id))
                .to_str()
                .unwrap()
                .to_string();
            let result = create_sst_file_writer::<E>(engine, cf, &path);
            match result {
                Ok(new_sst_writer) => {
                    let old_writer = sst_writer.replace(new_sst_writer);
                    let sst_info =
                        box_try!(finish_sst_writer(old_writer, prev_path, key_mgr.clone()));
                    stats.total_sst_size += sst_info.num_entries as usize;
                    generated_ssts_info.push(sst_info);
                }
                Err(e) => {
                    let io_error = io::Error::new(io::ErrorKind::Other, e);
                    return Err(io_error.into());
                }
            }
        }

        stats.key_count += 1;
        stats.total_kv_size += entry_len;

        if stats.total_kv_size >= next_io_check_size {
            handle_read_io_usage(&mut io_tracker, &mut remained_quota);
            next_io_check_size = stats.total_kv_size + SCAN_BYTES_PER_IO_LIMIT_CHECK;
        }

        if let Err(e) = sst_writer.borrow_mut().put(key, value) {
            let io_error = io::Error::new(io::ErrorKind::Other, e);
            return Err(io_error.into());
        }
        file_length += entry_len;
        Ok(true)
    };

    // Handle non-lmax
    scan_helper(iter, start_key, scan_fn);
    println!("scanned key_count: {}, ", stats.key_count);
    handle_read_io_usage(&mut io_tracker, &mut remained_quota);
    if stats.key_count > 0 {
        let final_sst_info = box_try!(finish_sst_writer(sst_writer.into_inner(), path, key_mgr));
        stats.total_sst_size += final_sst_info.num_entries as usize;
        cf_file.add_file(file_id);
        generated_ssts_info.push(final_sst_info);
    }

    info!(
        "build_sst_cf_file_list builds {} files in cf {}. Total keys {}, total kv size {}, total sst size {}. raw_size_per_file {}, total takes {:?}",
        file_id + 1,
        cf,
        stats.key_count,
        stats.total_kv_size,
        stats.total_sst_size,
        raw_size_per_file,
        instant.saturating_elapsed(),
    );

    if direct_ssts_info.is_some() {
        // HACK(XXX): Used to indicate we have generated some data.
        stats.key_count = 1;
    }

    Ok((
        stats,
        direct_ssts_info.unwrap_or_default(),
        generated_ssts_info,
    ))
}

/// Build a snapshot file for the given column family in sst format.
/// If there are no key-value pairs fetched, no files will be created at `path`,
/// otherwise the file will be created and synchronized.
pub fn build_sst_cf_file_list<E>(
    cf_file: &mut CfFile,
    engine: &E,
    snap: &E::Snapshot,
    start_key: &[u8],
    end_key: &[u8],
    raw_size_per_file: u64,
    io_limiter: &Limiter,
    key_mgr: Option<Arc<DataKeyManager>>,
    for_balance: bool,
) -> Result<(BuildStatistics, Vec<SstFileInfo>), Error>
where
    E: KvEngine,
{
    let cf = cf_file.cf;
    let mut stats = BuildStatistics::default();
    let mut sst_file_infos = Vec::new(); // 用于存储 SstFileInfo
    let mut remained_quota = 0;
    let mut file_id: usize = 0;
    let mut path = cf_file
        .path
        .join(cf_file.gen_tmp_file_name(file_id))
        .to_str()
        .unwrap()
        .to_string();
    let sst_writer = RefCell::new(create_sst_file_writer::<E>(engine, cf, &path)?);
    let mut file_length: usize = 0;

    let finish_sst_writer = |sst_writer: E::SstWriter,
                             path: String,
                             key_mgr: Option<Arc<DataKeyManager>>|
     -> Result<SstFileInfo, Error> {
        let info = sst_writer.finish()?;
        (|| {
            fail_point!("inject_sst_file_corruption", |_| {
                static CALLED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if CALLED
                    .compare_exchange(
                        false,
                        true,
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                    )
                    .is_err()
                {
                    return;
                }
                // overwrite the file to break checksum
                let mut f = OpenOptions::new().write(true).open(&path).unwrap();
                f.write_all(b"x").unwrap();
            });
        })();

        let sst_reader = E::SstReader::open(&path, key_mgr)?;
        if let Err(e) = sst_reader.verify_checksum() {
            fs::remove_file(&path)?;
            error!(
                "failed to pass block checksum verification";
                "file" => path,
                "err" => ?e,
            );
            return Err(io::Error::new(io::ErrorKind::InvalidData, e).into());
        }
        File::open(&path).and_then(|f| f.sync_all())?;

        // 从 info 中提取信息，构造 SstFileInfo
        let sst_file_info = SstFileInfo {
            file_name: path.clone(),
            smallest_key: info.smallest_key().to_vec(),
            largest_key: info.largest_key().to_vec(),
            num_entries: info.num_entries(),
        };
        Ok(sst_file_info)
    };

    let instant = Instant::now();
    let _io_type_guard = WithIoType::new(if for_balance {
        IoType::LoadBalance
    } else {
        IoType::Replication
    });

    let mut io_tracker = IoBytesTracker::new();
    let mut next_io_check_size = stats.total_kv_size + SCAN_BYTES_PER_IO_LIMIT_CHECK;
    let handle_read_io_usage = |io_tracker: &mut IoBytesTracker, remained_quota: &mut usize| {
        if let Some(io_bytes_delta) = io_tracker.update() {
            while io_bytes_delta.read as usize > *remained_quota {
                io_limiter.blocking_consume(IO_LIMITER_CHUNK_SIZE);
                *remained_quota += IO_LIMITER_CHUNK_SIZE;
            }
            *remained_quota -= io_bytes_delta.read as usize;
        }
    };

    box_try!(snap.scan(cf, start_key, end_key, false, |key, value| {
        let entry_len = key.len() + value.len();
        if file_length + entry_len > raw_size_per_file as usize {
            cf_file.add_file(file_id); // add previous file
            file_length = 0;
            file_id += 1;
            let prev_path = path.clone();
            path = cf_file
                .path
                .join(cf_file.gen_tmp_file_name(file_id))
                .to_str()
                .unwrap()
                .to_string();
            let result = create_sst_file_writer::<E>(engine, cf, &path);
            match result {
                Ok(new_sst_writer) => {
                    let old_writer = sst_writer.replace(new_sst_writer);
                    let sst_info =
                        box_try!(finish_sst_writer(old_writer, prev_path, key_mgr.clone()));
                    stats.total_sst_size += sst_info.num_entries as usize;
                    sst_file_infos.push(sst_info);
                }
                Err(e) => {
                    let io_error = io::Error::new(io::ErrorKind::Other, e);
                    return Err(io_error.into());
                }
            }
        }

        stats.key_count += 1;
        stats.total_kv_size += entry_len;

        if stats.total_kv_size >= next_io_check_size {
            handle_read_io_usage(&mut io_tracker, &mut remained_quota);
            next_io_check_size = stats.total_kv_size + SCAN_BYTES_PER_IO_LIMIT_CHECK;
        }

        if let Err(e) = sst_writer.borrow_mut().put(key, value) {
            let io_error = io::Error::new(io::ErrorKind::Other, e);
            return Err(io_error.into());
        }
        file_length += entry_len;
        Ok(true)
    }));

    handle_read_io_usage(&mut io_tracker, &mut remained_quota);

    if stats.key_count > 0 {
        let final_sst_info = box_try!(finish_sst_writer(sst_writer.into_inner(), path, key_mgr));
        stats.total_sst_size += final_sst_info.num_entries as usize;
        cf_file.add_file(file_id);
        sst_file_infos.push(final_sst_info);
    }

    info!(
        "build_sst_cf_file_list builds {} files in cf {}. Total keys {}, total kv size {}, total sst size {}. raw_size_per_file {}, total takes {:?}",
        file_id + 1,
        cf,
        stats.key_count,
        stats.total_kv_size,
        stats.total_sst_size,
        raw_size_per_file,
        instant.saturating_elapsed(),
    );

    Ok((stats, sst_file_infos))
}

/// Apply the given snapshot file into a column family. `callback` will be
/// invoked after each batch of key value pairs written to db.
///
/// Attention, callers should manually flush and sync the column family after
/// applying all sst files to make sure the data durability.
pub fn apply_plain_cf_file<E, F>(
    path: &str,
    key_mgr: Option<&Arc<DataKeyManager>>,
    stale_detector: &impl StaleDetector,
    db: &E,
    cf: &str,
    batch_size: usize,
    callback: &mut F,
) -> Result<(), Error>
where
    E: KvEngine,
    F: for<'r> FnMut(&'r [(Vec<u8>, Vec<u8>)]),
{
    let file = box_try!(File::open(path));
    let mut decoder = if let Some(key_mgr) = key_mgr {
        let reader = get_decrypter_reader(path, key_mgr)?;
        BufReader::new(reader)
    } else {
        BufReader::new(Box::new(file) as Box<dyn Read + Send>)
    };

    let mut wb = db.write_batch();
    let mut write_to_db = |batch: &mut Vec<(Vec<u8>, Vec<u8>)>| -> Result<(), EngineError> {
        batch.iter().try_for_each(|(k, v)| wb.put_cf(cf, k, v))?;
        wb.write()?;
        wb.clear();
        callback(batch);
        batch.clear();
        Ok(())
    };

    // Collect keys to a vec rather than wb so that we can invoke the callback less
    // times.
    let mut batch = Vec::with_capacity(1024);
    let mut batch_data_size = 0;

    loop {
        if stale_detector.is_stale() {
            return Err(Error::Abort);
        }
        let key = box_try!(decoder.decode_compact_bytes());
        if key.is_empty() {
            if !batch.is_empty() {
                box_try!(write_to_db(&mut batch));
            }
            return Ok(());
        }
        let value = box_try!(decoder.decode_compact_bytes());
        batch_data_size += key.len() + value.len();
        batch.push((key, value));
        if batch_data_size >= batch_size {
            box_try!(write_to_db(&mut batch));
            batch_data_size = 0;
        }
    }
}

pub fn apply_sst_cf_files_by_ingest<E>(files: &[&str], db: &E, cf: &str) -> Result<(), Error>
where
    E: KvEngine,
{
    if files.len() > 1 {
        info!(
            "apply_sst_cf_files_by_ingest starts on cf {}. All files {:?}",
            cf, files
        );
    }
    box_try!(db.ingest_external_file_cf(cf, files));
    Ok(())
}

fn apply_sst_cf_file_without_ingest<E, F>(
    path: &str,
    db: &E,
    cf: &str,
    key_mgr: Option<Arc<DataKeyManager>>,
    stale_detector: &impl StaleDetector,
    batch_size: usize,
    callback: &mut F,
) -> Result<(), Error>
where
    E: KvEngine,
    F: for<'r> FnMut(&'r [(Vec<u8>, Vec<u8>)]),
{
    let sst_reader = E::SstReader::open(path, key_mgr)?;
    let mut iter = sst_reader.iter(IterOptions::default())?;
    iter.seek_to_first()?;

    let mut wb = db.write_batch();
    let mut write_to_db = |batch: &mut Vec<(Vec<u8>, Vec<u8>)>| -> Result<(), EngineError> {
        batch.iter().try_for_each(|(k, v)| wb.put_cf(cf, k, v))?;
        wb.write()?;
        wb.clear();
        callback(batch);
        batch.clear();
        Ok(())
    };

    // Collect keys to a vec rather than wb so that we can invoke the callback less
    // times.
    let mut batch = Vec::with_capacity(1024);
    let mut batch_data_size = 0;
    loop {
        if stale_detector.is_stale() {
            return Err(Error::Abort);
        }
        if !iter.valid()? {
            break;
        }
        let key = iter.key().to_vec();
        let value = iter.value().to_vec();
        batch_data_size += key.len() + value.len();
        batch.push((key, value));
        if batch_data_size >= batch_size {
            box_try!(write_to_db(&mut batch));
            batch_data_size = 0;
        }
        iter.next()?;
    }
    if !batch.is_empty() {
        box_try!(write_to_db(&mut batch));
    }
    Ok(())
}

/// Apply the given snapshot file into a column family by directly writing kv
/// pairs to db, without ingesting them. `callback` will be invoked after each
/// batch of key value pairs written to db.
///
/// Attention, callers should manually flush and sync the column family after
/// applying all sst files to make sure the data durability.
pub fn apply_sst_cf_files_without_ingest<E, F>(
    files: &[&str],
    db: &E,
    cf: &str,
    key_mgr: Option<Arc<DataKeyManager>>,
    stale_detector: &impl StaleDetector,
    batch_size: usize,
    callback: &mut F,
) -> Result<(), Error>
where
    E: KvEngine,
    F: for<'r> FnMut(&'r [(Vec<u8>, Vec<u8>)]),
{
    for path in files {
        box_try!(apply_sst_cf_file_without_ingest(
            path,
            db,
            cf,
            key_mgr.clone(),
            stale_detector,
            batch_size,
            callback
        ));
    }
    Ok(())
}

fn create_sst_file_writer<E>(engine: &E, cf: CfName, path: &str) -> Result<E::SstWriter, Error>
where
    E: KvEngine,
{
    let builder = E::SstWriterBuilder::new()
        .set_db(engine)
        .set_cf(cf)
        .set_compression_type(Some(SstCompressionType::Zstd));
    let writer = box_try!(builder.build(path));
    Ok(writer)
}

// TODO: Use DataKeyManager::open_file_for_read() instead.
pub fn get_decrypter_reader(
    file: &str,
    encryption_key_manager: &DataKeyManager,
) -> Result<Box<dyn Read + Send>, Error> {
    let enc_info = box_try!(encryption_key_manager.get_file(file));
    let mthd = enc_info.method;
    debug!(
        "get_decrypter_reader gets enc_info for {:?}, method: {:?}",
        file, mthd
    );
    if mthd == EncryptionMethod::Plaintext {
        let f = box_try!(File::open(file));
        return Ok(Box::new(f) as Box<dyn Read + Send>);
    }
    let iv = box_try!(Iv::from_slice(&enc_info.iv));
    let f = box_try!(File::open(file));
    let r = box_try!(DecrypterReader::new(f, mthd, &enc_info.key, iv));
    Ok(Box::new(r) as Box<dyn Read + Send>)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, iter::Iterator, path::PathBuf};

    use engine_rocks::{
        raw::{BlockBasedOptions, DBCompressionType},
        util::new_engine_opt,
        RocksCfOptions, RocksDbOptions, RocksEngine, RocksSstPartitionerFactory,
    };
    use engine_test::kv::KvTestEngine;
    use engine_traits::{CompactExt, MiscExt, SyncMutable, CF_DEFAULT};
    use kvproto::metapb::Region;
    use tempfile::{Builder, TempDir};
    use tikv_util::time::Limiter;

    use super::*;
    use crate::{
        coprocessor::region_info_accessor::MockRegionInfoProvider,
        store::{
            snap::{tests::*, SNAPSHOT_CFS, SST_FILE_SUFFIX},
            CompactionGuardGeneratorFactory,
        },
    };

    struct TestStaleDetector;
    impl StaleDetector for TestStaleDetector {
        fn is_stale(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_cf_build_and_apply_plain_files() {
        let db_creaters = &[open_test_empty_db, open_test_db];
        for db_creater in db_creaters {
            let (_enc_dir, enc_opts) =
                gen_db_options_with_encryption("test_cf_build_and_apply_plain_files_enc");
            for db_opt in [None, Some(enc_opts)] {
                let dir = Builder::new().prefix("test-snap-cf-db").tempdir().unwrap();
                let db: KvTestEngine = db_creater(dir.path(), db_opt.clone(), None).unwrap();
                // Collect keys via the key_callback into a collection.
                let mut applied_keys: HashMap<_, Vec<_>> = HashMap::new();
                let dir1 = Builder::new()
                    .prefix("test-snap-cf-db-apply")
                    .tempdir()
                    .unwrap();
                let db1: KvTestEngine = open_test_empty_db(dir1.path(), db_opt, None).unwrap();

                let snap = db.snapshot();
                for cf in SNAPSHOT_CFS {
                    let snap_cf_dir = Builder::new().prefix("test-snap-cf").tempdir().unwrap();
                    let mut cf_file = CfFile {
                        cf,
                        path: PathBuf::from(snap_cf_dir.path().to_str().unwrap()),
                        file_prefix: "test_plain_sst".to_string(),
                        file_suffix: SST_FILE_SUFFIX.to_string(),
                        ..Default::default()
                    };
                    let stats = build_plain_cf_file::<KvTestEngine>(
                        &mut cf_file,
                        None,
                        &snap,
                        &keys::data_key(b"a"),
                        &keys::data_end_key(b"z"),
                    )
                    .unwrap();
                    if stats.key_count == 0 {
                        assert_eq!(cf_file.file_paths().len(), 0);
                        assert_eq!(cf_file.clone_file_paths().len(), 0);
                        assert_eq!(cf_file.tmp_file_paths().len(), 0);
                        assert_eq!(cf_file.size.len(), 0);
                        continue;
                    }

                    let detector = TestStaleDetector {};
                    let tmp_file_path = &cf_file.tmp_file_paths()[0];
                    apply_plain_cf_file(
                        tmp_file_path,
                        None,
                        &detector,
                        &db1,
                        cf,
                        16,
                        &mut |v: &[(Vec<u8>, Vec<u8>)]| {
                            v.iter()
                                .cloned()
                                .for_each(|pair| applied_keys.entry(cf).or_default().push(pair))
                        },
                    )
                    .unwrap();
                }

                assert_eq_db(&db, &db1);

                // Scan keys from db
                let mut keys_in_db: HashMap<_, Vec<_>> = HashMap::new();
                for cf in SNAPSHOT_CFS {
                    snap.scan(
                        cf,
                        &keys::data_key(b"a"),
                        &keys::data_end_key(b"z"),
                        true,
                        |k, v| {
                            keys_in_db
                                .entry(cf)
                                .or_default()
                                .push((k.to_owned(), v.to_owned()));
                            Ok(true)
                        },
                    )
                    .unwrap();
                }
                assert_eq!(applied_keys, keys_in_db);
            }
        }
    }

    #[test]
    fn test_cf_build_and_apply_sst_files() {
        let db_creaters = &[open_test_empty_db, open_test_db_with_100keys];
        let max_file_sizes = &[u64::MAX, 100];
        let limiter = Limiter::new(f64::INFINITY);
        for max_file_size in max_file_sizes {
            for db_creater in db_creaters {
                let (_enc_dir, enc_opts) =
                    gen_db_options_with_encryption("test_cf_build_and_apply_sst_files_enc");
                for db_opt in [None, Some(enc_opts)] {
                    let dir = Builder::new().prefix("test-snap-cf-db").tempdir().unwrap();
                    let db = db_creater(dir.path(), db_opt.clone(), None).unwrap();
                    let snap_cf_dir = Builder::new().prefix("test-snap-cf").tempdir().unwrap();
                    let mut cf_file = CfFile {
                        cf: CF_DEFAULT,
                        path: PathBuf::from(snap_cf_dir.path().to_str().unwrap()),
                        file_prefix: "test_sst".to_string(),
                        file_suffix: SST_FILE_SUFFIX.to_string(),
                        ..Default::default()
                    };
                    let stats = build_sst_cf_file_list::<KvTestEngine>(
                        &mut cf_file,
                        &db,
                        &db.snapshot(),
                        &keys::data_key(b"a"),
                        &keys::data_key(b"z"),
                        *max_file_size,
                        &limiter,
                        db_opt.as_ref().and_then(|opt| opt.get_key_manager()),
                        true,
                    )
                    .unwrap()
                    .0;
                    if stats.key_count == 0 {
                        assert_eq!(cf_file.file_paths().len(), 0);
                        assert_eq!(cf_file.clone_file_paths().len(), 0);
                        assert_eq!(cf_file.tmp_file_paths().len(), 0);
                        assert_eq!(cf_file.size.len(), 0);
                        assert_eq!(cf_file.checksum.len(), 0);
                        continue;
                    } else {
                        assert!(
                            cf_file.file_paths().len() == 12 && *max_file_size < u64::MAX
                                || cf_file.file_paths().len() == 1 && *max_file_size == u64::MAX
                        );
                        assert!(cf_file.clone_file_paths().len() == cf_file.file_paths().len());
                        assert!(cf_file.tmp_file_paths().len() == cf_file.file_paths().len());
                        assert!(cf_file.size.len() == cf_file.file_paths().len());
                        assert!(cf_file.checksum.len() == cf_file.file_paths().len());
                    }

                    let dir1 = Builder::new()
                        .prefix("test-snap-cf-db-apply")
                        .tempdir()
                        .unwrap();
                    let db1: KvTestEngine = open_test_empty_db(dir1.path(), db_opt, None).unwrap();
                    let tmp_file_paths = cf_file.tmp_file_paths();
                    let tmp_file_paths = tmp_file_paths
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<&str>>();
                    apply_sst_cf_files_by_ingest(&tmp_file_paths, &db1, CF_DEFAULT).unwrap();
                    assert_eq_db(&db, &db1);
                }
            }
        }
    }

    // This test verifies that building SST files is effectively limited by the I/O
    // limiter based on actual I/O usage. It achieve this by adding an I/O limiter
    // and asserting that the elapsed time for building SST files exceeds the
    // lower bound enforced by the I/O limiter.
    //
    // In this test, the I/O limiter is configured with a throughput limit 8000
    // bytes/sec. A dataset of 1000 keys (totaling 11, 890 bytes) is generated  to
    // trigger two I/O limiter checks, as the default SCAN_BYTES_PER_IO_LIMIT_CHECK
    // is 8192 bytes. During each check, the mocked `get_thread_io_bytes_stats`
    // function returns 4096 bytes of I/O usage, resulting in total of 8192 bytes.
    // With the 8000 bytes/sec limitation, we assert that the elapsed time must
    // exceed 1 second.
    #[test]
    fn test_build_sst_with_io_limiter() {
        #[cfg(not(feature = "failpoints"))]
        return;

        let dir = Builder::new().prefix("test-io-limiter").tempdir().unwrap();
        let db = open_test_db_with_nkeys(dir.path(), None, None, 1000).unwrap();
        println!("path: {:?}", dir.path());
        // The max throughput is 8000 bytes/sec.
        let bytes_per_sec = 8000_f64;
        let limiter = Limiter::new(bytes_per_sec);
        let snap_dir = Builder::new().prefix("snap-dir").tempdir().unwrap();
        let mut cf_file = CfFile {
            cf: CF_DEFAULT,
            path: PathBuf::from(snap_dir.path()),
            file_prefix: "test_sst".to_string(),
            file_suffix: SST_FILE_SUFFIX.to_string(),
            ..Default::default()
        };

        let start = Instant::now();
        fail::cfg("delta_read_io_bytes", "return(4096)").unwrap();
        let stats = build_sst_cf_file_list::<KvTestEngine>(
            &mut cf_file,
            &db,
            &db.snapshot(),
            &keys::data_key(b""),
            &keys::data_key(b"z"),
            u64::MAX,
            &limiter,
            None,
            true,
        )
        .unwrap()
        .0;
        assert_eq!(stats.total_kv_size, 11890);
        // Must exceed 1 second!
        assert!(start.saturating_elapsed_secs() > 1_f64);
    }

    fn print_sst_files_info(sst_file_infos: &Vec<SstFileInfo>) {
        for sst_file in sst_file_infos {
            println!(
                "Generated SST file for region {}: file_name = {}, smallest_key = {}, largest_key = {}, num_entries = {}",
                1,
                sst_file.file_name,
                String::from_utf8_lossy(&sst_file.smallest_key), // 转换为字符串
                String::from_utf8_lossy(&sst_file.largest_key),  // 转换为字符串
                sst_file.num_entries,
            );
        }
    }

    fn level_files(db: &RocksEngine) -> collections::HashMap<usize, Vec<FileMetadata>> {
        let db = db.as_inner();
        let cf = db.cf_handle("default").unwrap();
        let md = db.get_column_family_meta_data(cf);
        let mut res: collections::HashMap<usize, Vec<FileMetadata>> =
            collections::HashMap::default();
        for (i, level) in md.get_levels().into_iter().enumerate() {
            for file in level.get_files() {
                res.entry(i).or_default().push(FileMetadata {
                    name: file.get_name(),
                    size: file.get_size(),
                    smallest_key: file.get_smallestkey().to_owned(),
                    largest_key: file.get_largestkey().to_owned(),
                });
            }
        }
        res
    }

    fn new_test_db(provider: MockRegionInfoProvider) -> (RocksEngine, TempDir) {
        let temp_dir = Builder::new()
            .prefix("test-build-lmax-sst")
            .tempdir_in("/tmp")
            .unwrap();
        println!("Temporary directory created: {:?}", temp_dir);

        const MIN_OUTPUT_FILE_SIZE: u64 = 1024 * 10;
        const MAX_OUTPUT_FILE_SIZE: u64 = 1024 * 11;
        const MAX_COMPACTION_SIZE: u64 = u64::MAX;
        let mut cf_opts = RocksCfOptions::default();
        cf_opts.set_max_bytes_for_level_base(MAX_OUTPUT_FILE_SIZE);
        cf_opts.set_max_bytes_for_level_multiplier(5);
        cf_opts.set_target_file_size_base(MAX_OUTPUT_FILE_SIZE);
        cf_opts.set_level_compaction_dynamic_level_bytes(false);
        cf_opts.set_sst_partitioner_factory(RocksSstPartitionerFactory(
            CompactionGuardGeneratorFactory::new(
                CF_DEFAULT,
                provider,
                MIN_OUTPUT_FILE_SIZE,
                MAX_COMPACTION_SIZE,
            )
            .unwrap(),
        ));
        cf_opts.set_disable_auto_compactions(true);
        cf_opts.compression_per_level(&[
            DBCompressionType::No,
            DBCompressionType::No,
            DBCompressionType::No,
            DBCompressionType::No,
            DBCompressionType::No,
            DBCompressionType::No,
            DBCompressionType::No,
        ]);
        // Make block size small to make sure current_output_file_size passed to
        // SstPartitioner is accurate.
        let mut block_based_opts = BlockBasedOptions::new();
        block_based_opts.set_block_size(100);
        cf_opts.set_block_based_table_factory(&block_based_opts);

        let db = new_engine_opt(
            temp_dir.path().to_str().unwrap(),
            RocksDbOptions::default(),
            vec![(CF_DEFAULT, cf_opts)],
        )
        .unwrap();
        (db, temp_dir)
    }

    // region 1, lmax 有数据，non-lmax 也有数据，lmax 独占文件
    // region 2, lmax 有数据，non-lmax 没数据，lmax 独占文件
    // region 3, lmax 有数据，non-lmax 没数据，lmax 只有一个文件，且共享
    // region 4, lmax 有数据，non-lmax 有数据，lmax 共享文件
    // region 5, lmax 有数据，non-lmax 没数据，lmax 有多个文件，其中一个共享
    // region 6, lmax 没数据，non-lmax 有数据
    #[test]
    fn test_build_lmax_sst() {
        let provider = MockRegionInfoProvider::new(vec![
            Region {
                id: 1,
                start_key: b"a".to_vec(),
                end_key: b"b".to_vec(),
                ..Default::default()
            },
            Region {
                id: 2,
                start_key: b"b".to_vec(),
                end_key: b"c".to_vec(),
                ..Default::default()
            },
            Region {
                id: 3,
                start_key: b"c".to_vec(),
                end_key: b"d".to_vec(),
                ..Default::default()
            },
            Region {
                id: 4,
                start_key: b"d".to_vec(),
                end_key: b"e".to_vec(),
                ..Default::default()
            },
            Region {
                id: 5,
                start_key: b"e".to_vec(),
                end_key: b"f".to_vec(),
                ..Default::default()
            },
            Region {
                id: 6,
                start_key: b"f".to_vec(),
                end_key: b"g".to_vec(),
                ..Default::default()
            },
        ]);
        let (db, dir) = new_test_db(provider);

        {
            let value = vec![b'v'; 1000];
            // region 1, lmax 有数据，non-lmax 也有数据，lmax 独占文件两个文件
            {
                ['a']
                    .into_iter()
                    .flat_map(|x| (1..=15).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &value).unwrap());
                db.flush_cfs(&[], true).unwrap();
                db.compact_files_in_range(None, None, Some(2)).unwrap();
            }

            {
                ['b']
                    .into_iter()
                    .flat_map(|x| (1..=10).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &value).unwrap());
                db.flush_cfs(&[], true).unwrap();
                db.compact_files_in_range(None, None, Some(2)).unwrap();
            }

            {
                ['c']
                    .into_iter()
                    .flat_map(|x| (1..=3).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &value).unwrap());
                db.flush_cfs(&[], true).unwrap();
            }

            {
                ['d']
                    .into_iter()
                    // 13 是为了和 c?? 以及 e?? 共享一个文件，然后自己再独占一个文件
                    .flat_map(|x| (1..=23).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &value).unwrap());
                db.flush_cfs(&[], true).unwrap();
            }

            {
                ['e']
                    .into_iter()
                    // 13 是为了和 c?? 以及 e?? 共享一个文件，然后自己再独占一个文件
                    .flat_map(|x| (1..=13).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &value).unwrap());
                db.flush_cfs(&[], true).unwrap();
            }
            db.compact_files_in_range(None, None, Some(2)).unwrap();

            /////////////// no -lmax
            let tiny_value = [b'v'; 1];
            {
                ['a']
                    .into_iter()
                    .flat_map(|x| (1..=10).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &tiny_value).unwrap());
                db.flush_cfs(&[], true).unwrap();
                db.compact_files_in_range(None, None, Some(1)).unwrap();
            }

            {
                ['e']
                    .into_iter()
                    .flat_map(|x| (1..=10).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &tiny_value).unwrap());
                db.flush_cfs(&[], true).unwrap();
                db.compact_files_in_range(None, None, Some(1)).unwrap();
            }

            {
                ['f']
                    .into_iter()
                    .flat_map(|x| (1..=10).map(move |n| format!("z{x}{:02}", n).into_bytes()))
                    .for_each(|key| db.put(&key, &tiny_value).unwrap());
                db.flush_cfs(&[], true).unwrap();
                db.compact_files_in_range(None, None, Some(1)).unwrap();
            }

            let level_2 = &level_files(&db)[&2];
            assert_eq!(level_2.len(), 7);
            let level_1 = &level_files(&db)[&1];
            assert_eq!(level_1.len(), 3);
        }

        // Test region1
        {
            let limiter = Limiter::new(f64::INFINITY);
            let snap_dir = Builder::new()
                .prefix("snap-dir")
                .tempdir_in("/tmp")
                .unwrap();
            println!("Snap directory created: {:?}", snap_dir);
            let mut cf_file = CfFile {
                cf: CF_DEFAULT,
                path: PathBuf::from(snap_dir.path()),
                file_prefix: "test_sst".to_string(),
                file_suffix: SST_FILE_SUFFIX.to_string(),
                ..Default::default()
            };

            let (stats, filter_files, sst_file_infos) = build_sst_cf_file_list_new::<KvTestEngine>(
                &mut cf_file,
                &db,
                &db.snapshot(),
                &keys::data_key(b"a"),
                &keys::data_key(b"b"),
                u64::MAX,
                &limiter,
                None,
                true,
            )
            .unwrap();
            assert_eq!(filter_files.len(), 2);
            assert_eq!(sst_file_infos.len(), 1);
            assert_eq!(
                filter_files[0],
                FileMetadata {
                    name: "test_sst.sst.tmp".to_string(),
                    size: 11423,
                    smallest_key: "za01".as_bytes().to_vec(),
                    largest_key: "za10".as_bytes().to_vec(),
                }
            );
            assert_eq!(
                filter_files[1],
                FileMetadata {
                    name: "test_sst_0001.sst.tmp".to_string(),
                    size: 6208,
                    smallest_key: "za11".as_bytes().to_vec(),
                    largest_key: "za15".as_bytes().to_vec(),
                }
            );
            assert_eq!(
                sst_file_infos[0],
                SstFileInfo {
                    file_name: "test_sst_0002.sst.tmp".to_string(),
                    smallest_key: "za01".as_bytes().to_vec(),
                    largest_key: "za10".as_bytes().to_vec(),
                    num_entries: 10,
                }
            );
        }

        // Test region2
        {
            let limiter = Limiter::new(f64::INFINITY);
            let snap_dir = Builder::new()
                .prefix("snap-dir")
                .tempdir_in("/tmp")
                .unwrap();
            println!("Snap directory created: {:?}", snap_dir);
            let mut cf_file = CfFile {
                cf: CF_DEFAULT,
                path: PathBuf::from(snap_dir.path()),
                file_prefix: "test_sst".to_string(),
                file_suffix: SST_FILE_SUFFIX.to_string(),
                ..Default::default()
            };

            let (stats, filter_files, sst_file_infos) = build_sst_cf_file_list_new::<KvTestEngine>(
                &mut cf_file,
                &db,
                &db.snapshot(),
                &keys::data_key(b"b"),
                &keys::data_key(b"c"),
                u64::MAX,
                &limiter,
                None,
                true,
            )
            .unwrap();
            print_sst_files_info(&sst_file_infos);
            assert_eq!(filter_files.len(), 1);
            assert_eq!(sst_file_infos.len(), 0);
            assert_eq!(
                filter_files[0],
                FileMetadata {
                    name: "test_sst.sst.tmp".to_string(),
                    size: 11423,
                    smallest_key: "zb01".as_bytes().to_vec(),
                    largest_key: "zb10".as_bytes().to_vec(),
                }
            );
        }

        // Test region3
        {
            let limiter = Limiter::new(f64::INFINITY);
            let snap_dir = Builder::new()
                .prefix("snap-dir")
                .tempdir_in("/tmp")
                .unwrap();
            println!("Snap directory created: {:?}", snap_dir);
            let mut cf_file = CfFile {
                cf: CF_DEFAULT,
                path: PathBuf::from(snap_dir.path()),
                file_prefix: "test_sst".to_string(),
                file_suffix: SST_FILE_SUFFIX.to_string(),
                ..Default::default()
            };

            let (stats, filter_files, sst_file_infos) = build_sst_cf_file_list_new::<KvTestEngine>(
                &mut cf_file,
                &db,
                &db.snapshot(),
                &keys::data_key(b"c"),
                &keys::data_key(b"d"),
                u64::MAX,
                &limiter,
                None,
                true,
            )
            .unwrap();
            print_sst_files_info(&sst_file_infos);
            assert_eq!(filter_files.len(), 0);
            assert_eq!(sst_file_infos.len(), 1);
            assert_eq!(
                sst_file_infos[0],
                SstFileInfo {
                    file_name: "test_sst.sst.tmp".to_string(),
                    smallest_key: "zc01".as_bytes().to_vec(),
                    largest_key: "zc03".as_bytes().to_vec(),
                    num_entries: 3,
                }
            );
        }

        // Test region4
        {
            let limiter = Limiter::new(f64::INFINITY);
            let snap_dir = Builder::new()
                .prefix("snap-dir")
                .tempdir_in("/tmp")
                .unwrap();
            println!("Snap directory created: {:?}", snap_dir);
            let mut cf_file = CfFile {
                cf: CF_DEFAULT,
                path: PathBuf::from(snap_dir.path()),
                file_prefix: "test_sst".to_string(),
                file_suffix: SST_FILE_SUFFIX.to_string(),
                ..Default::default()
            };

            let (stats, filter_files, sst_file_infos) = build_sst_cf_file_list_new::<KvTestEngine>(
                &mut cf_file,
                &db,
                &db.snapshot(),
                &keys::data_key(b"d"),
                &keys::data_key(b"e"),
                u64::MAX,
                &limiter,
                None,
                true,
            )
            .unwrap();
            print_sst_files_info(&sst_file_infos);
            assert_eq!(filter_files.len(), 0);
            assert_eq!(sst_file_infos.len(), 1);
            assert_eq!(
                sst_file_infos[0],
                SstFileInfo {
                    file_name: "test_sst.sst.tmp".to_string(),
                    smallest_key: "zd01".as_bytes().to_vec(),
                    largest_key: "zd23".as_bytes().to_vec(),
                    num_entries: 23,
                }
            );
        }

        // Test region5
        {
            let limiter = Limiter::new(f64::INFINITY);
            let snap_dir = Builder::new()
                .prefix("snap-dir")
                .tempdir_in("/tmp")
                .unwrap();
            println!("Snap directory created: {:?}", snap_dir);
            let mut cf_file = CfFile {
                cf: CF_DEFAULT,
                path: PathBuf::from(snap_dir.path()),
                file_prefix: "test_sst".to_string(),
                file_suffix: SST_FILE_SUFFIX.to_string(),
                ..Default::default()
            };

            let (stats, filter_files, sst_file_infos) = build_sst_cf_file_list_new::<KvTestEngine>(
                &mut cf_file,
                &db,
                &db.snapshot(),
                &keys::data_key(b"e"),
                &keys::data_key(b"f"),
                u64::MAX,
                &limiter,
                None,
                true,
            )
            .unwrap();
            print_sst_files_info(&sst_file_infos);
            assert_eq!(filter_files.len(), 0);
            assert_eq!(sst_file_infos.len(), 1);
            assert_eq!(
                sst_file_infos[0],
                SstFileInfo {
                    file_name: "test_sst.sst.tmp".to_string(),
                    smallest_key: "ze01".as_bytes().to_vec(),
                    largest_key: "ze13".as_bytes().to_vec(),
                    num_entries: 13,
                }
            );
        }

        // Test region6
        {
            let limiter = Limiter::new(f64::INFINITY);
            let snap_dir = Builder::new()
                .prefix("snap-dir")
                .tempdir_in("/tmp")
                .unwrap();
            println!("Snap directory created: {:?}", snap_dir);
            let mut cf_file = CfFile {
                cf: CF_DEFAULT,
                path: PathBuf::from(snap_dir.path()),
                file_prefix: "test_sst".to_string(),
                file_suffix: SST_FILE_SUFFIX.to_string(),
                ..Default::default()
            };

            let (stats, filter_files, sst_file_infos) = build_sst_cf_file_list_new::<KvTestEngine>(
                &mut cf_file,
                &db,
                &db.snapshot(),
                &keys::data_key(b"f"),
                &keys::data_key(b"g"),
                u64::MAX,
                &limiter,
                None,
                true,
            )
            .unwrap();
            print_sst_files_info(&sst_file_infos);
            assert_eq!(filter_files.len(), 0);
            assert_eq!(sst_file_infos.len(), 1);
            assert_eq!(
                sst_file_infos[0],
                SstFileInfo {
                    file_name: "test_sst.sst.tmp".to_string(),
                    smallest_key: "zf01".as_bytes().to_vec(),
                    largest_key: "zf10".as_bytes().to_vec(),
                    num_entries: 10,
                }
            );
        }
    }
}
