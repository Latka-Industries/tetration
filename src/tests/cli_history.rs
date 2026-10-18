//! CLI query history (platform cache JSONL, not stored in `.tet`).

use std::env;
use std::sync::{Mutex, MutexGuard};

use crate::query::cli::{
    HistoryExecuteFilter, HistoryListFilter, HistorySettings, append_cli_query_history,
    clear_cli_query_history, cli_query_history_max, format_history_list_json,
    format_history_list_text, get_cli_query_history_entry, history_entry_mode,
    list_cli_query_history, parse_history_execute_filter,
};
use crate::query::{parse_query_json, validate_query};

static HISTORY_ENV_LOCK: Mutex<()> = Mutex::new(());

fn history_lock() -> MutexGuard<'static, ()> {
    HISTORY_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn with_history_file<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let _guard = history_lock();
    let temp = tempfile::NamedTempFile::new().unwrap();
    let path = temp.path().to_path_buf();
    // Test-only: isolate history path per test (Rust 2024: env mutators are unsafe).
    unsafe {
        env::set_var("TET_QUERY_HISTORY_FILE", &path);
        env::remove_var("TET_QUERY_HISTORY_MAX");
    }
    let out = f();
    unsafe {
        env::remove_var("TET_QUERY_HISTORY_FILE");
        env::remove_var("TET_QUERY_HISTORY_MAX");
    }
    out
}

fn sample_query_json() -> String {
    super::fixture::query_files::read("slice_full_temperature", "json")
}

fn mean_query_json() -> String {
    super::fixture::query_files::read("mean_temperature", "json")
}

#[test]
fn history_settings_default_and_env() {
    with_history_file(|| {
        let d = HistorySettings::default();
        assert_eq!(d.cli_query_max, 50);
        assert_eq!(d.history_max_cap, 10_000);
        assert_eq!(d.history_file_name, "query_history.jsonl");
        unsafe {
            env::set_var("TET_QUERY_HISTORY_MAX", "3");
        }
        let from_env = HistorySettings::from_env();
        assert_eq!(from_env.cli_query_max, 3);
        assert_eq!(cli_query_history_max(), 3);
    });
}

#[test]
fn cli_query_history_append_trim_and_list() {
    with_history_file(|| {
        unsafe {
            env::set_var("TET_QUERY_HISTORY_MAX", "10");
        }
        let max = cli_query_history_max();
        assert_eq!(max, 10);
        let doc = parse_query_json(&sample_query_json()).unwrap();
        validate_query(&doc).unwrap();
        for i in 0..12 {
            append_cli_query_history(&doc, Some(&format!("/data/{i}.tet")), i % 2 == 0).unwrap();
        }
        let listed = list_cli_query_history(max, false, None).unwrap();
        assert_eq!(listed.len(), max);
        assert_eq!(listed[0].tet.as_deref(), Some("/data/11.tet"));
        assert_eq!(listed[9].tet.as_deref(), Some("/data/2.tet"));
    });
}

#[test]
fn cli_query_history_dedupes_consecutive_duplicate() {
    with_history_file(|| {
        let doc = parse_query_json(&sample_query_json()).unwrap();
        append_cli_query_history(&doc, Some("a.tet"), true).unwrap();
        append_cli_query_history(&doc, Some("a.tet"), true).unwrap();
        append_cli_query_history(&doc, Some("a.tet"), true).unwrap();
        let listed = list_cli_query_history(10, true, None).unwrap();
        assert_eq!(listed.len(), 1);
        append_cli_query_history(&doc, Some("b.tet"), true).unwrap();
        assert_eq!(list_cli_query_history(10, true, None).unwrap().len(), 2);
    });
}

#[test]
fn cli_query_history_max_env_trims() {
    with_history_file(|| {
        unsafe {
            env::set_var("TET_QUERY_HISTORY_MAX", "3");
        }
        assert_eq!(cli_query_history_max(), 3);
        let doc = parse_query_json(&sample_query_json()).unwrap();
        for i in 0..5 {
            append_cli_query_history(&doc, Some(&format!("/data/{i}.tet")), false).unwrap();
        }
        let all = list_cli_query_history(100, true, None).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].tet.as_deref(), Some("/data/4.tet"));
        assert_eq!(all[2].tet.as_deref(), Some("/data/2.tet"));
    });
}

#[test]
fn format_history_list_text_shows_index_and_dataset() {
    with_history_file(|| {
        let doc = parse_query_json(&sample_query_json()).unwrap();
        append_cli_query_history(&doc, Some("target/demo/x.tet"), true).unwrap();
        let entries = list_cli_query_history(10, false, None).unwrap();
        let text = format_history_list_text(&entries, None, &HistorySettings::default(), None);
        assert!(text.contains("  1 "));
        assert!(text.contains("temperature"));
        assert!(text.contains("mode"));
        assert_eq!(history_entry_mode(true), "x");
        assert_eq!(history_entry_mode(false), "p");
        assert!(text.contains(" x "));
        assert!(text.contains("mode: x = had -x"));
        assert!(text.contains("replay:"));
        assert!(!text.contains("bookmark"));
        assert!(!text.contains("\"entries\""));
    });
}

#[test]
fn format_history_list_json_includes_mode() {
    with_history_file(|| {
        let doc = parse_query_json(&sample_query_json()).unwrap();
        append_cli_query_history(&doc, Some("a.tet"), true).unwrap();
        append_cli_query_history(&doc, None, false).unwrap();
        let entries = list_cli_query_history(10, true, None).unwrap();
        let json =
            format_history_list_json(&entries, None, &HistorySettings::default(), None).unwrap();
        assert!(json.contains(r#""mode": "x""#));
        assert!(json.contains(r#""mode": "p""#));
        assert!(json.contains("mode_key"));
        assert!(!json.contains(r#""execute""#));
    });
}

#[test]
fn cli_query_history_list_all_flag() {
    with_history_file(|| {
        let doc = parse_query_json(&sample_query_json()).unwrap();
        for i in 0..5 {
            append_cli_query_history(&doc, Some(&format!("/data/{i}.tet")), false).unwrap();
        }
        assert_eq!(list_cli_query_history(2, false, None).unwrap().len(), 2);
        assert_eq!(list_cli_query_history(2, true, None).unwrap().len(), 5);
    });
}

#[test]
fn cli_query_history_get_entry_by_index() {
    with_history_file(|| {
        let doc = parse_query_json(&sample_query_json()).unwrap();
        append_cli_query_history(&doc, Some("/a.tet"), true).unwrap();
        append_cli_query_history(&doc, Some("/b.tet"), false).unwrap();
        let newest = get_cli_query_history_entry(1, None).unwrap();
        assert_eq!(newest.tet.as_deref(), Some("/b.tet"));
        assert!(!newest.execute);
        let older = get_cli_query_history_entry(2, None).unwrap();
        assert_eq!(older.tet.as_deref(), Some("/a.tet"));
        assert!(older.execute);
        assert!(get_cli_query_history_entry(0, None).is_err());
        assert!(get_cli_query_history_entry(3, None).is_err());
    });
}

#[test]
fn cli_query_history_clear() {
    with_history_file(|| {
        let doc = parse_query_json(&sample_query_json()).unwrap();
        validate_query(&doc).unwrap();
        append_cli_query_history(&doc, None, false).unwrap();
        assert_eq!(list_cli_query_history(10, false, None).unwrap().len(), 1);
        clear_cli_query_history().unwrap();
        assert!(list_cli_query_history(10, false, None).unwrap().is_empty());
    });
}

#[test]
fn cli_query_history_disabled_by_env() {
    with_history_file(|| {
        unsafe {
            env::set_var("TET_NO_QUERY_HISTORY", "1");
        }
        let doc = parse_query_json(&sample_query_json()).unwrap();
        append_cli_query_history(&doc, None, false).unwrap();
        assert!(list_cli_query_history(10, false, None).unwrap().is_empty());
        unsafe {
            env::remove_var("TET_NO_QUERY_HISTORY");
        }
    });
}

#[test]
fn history_list_filter_dataset_and_mode() {
    with_history_file(|| {
        let temp = parse_query_json(&sample_query_json()).unwrap();
        let mean = parse_query_json(&mean_query_json()).unwrap();
        append_cli_query_history(&temp, Some("/cf_3d.tet"), true).unwrap();
        append_cli_query_history(&mean, Some("/tensor_3d.tet"), false).unwrap();
        append_cli_query_history(&temp, Some("/other.tet"), false).unwrap();

        let filter = HistoryListFilter {
            dataset: Some("temp".to_owned()),
            tet: None,
            mode: Some(HistoryExecuteFilter::Execute),
            grep: None,
        };
        let listed = list_cli_query_history(10, true, Some(&filter)).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].tet.as_deref(), Some("/cf_3d.tet"));
        assert!(listed[0].execute);

        let grep = HistoryListFilter {
            dataset: None,
            tet: None,
            mode: None,
            grep: Some("tensor".to_owned()),
        };
        assert_eq!(
            list_cli_query_history(10, true, Some(&grep)).unwrap().len(),
            1
        );

        let text =
            format_history_list_text(&listed, None, &HistorySettings::default(), Some(&filter));
        assert!(text.contains("filter: dataset~temp mode=x"));
    });
}

#[test]
fn parse_history_execute_filter_tokens() {
    assert_eq!(
        parse_history_execute_filter("x").unwrap(),
        HistoryExecuteFilter::Execute
    );
    assert_eq!(
        parse_history_execute_filter("plan").unwrap(),
        HistoryExecuteFilter::Plan
    );
    assert!(parse_history_execute_filter("bogus").is_err());
}
