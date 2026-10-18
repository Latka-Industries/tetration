//! Shared CLI helpers (`error:` prefix, query JSON input).

#![allow(deprecated)]

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use crate::args::QueryStdoutFormat;
use tetration::query::{QueryInputFormat, QueryOutputFormat, detect_query_input_format};

pub(crate) fn cli_error(message: impl std::fmt::Display) -> String {
    format!("error: {message}")
}

pub(crate) fn resolve_stdout(quiet: bool, format: QueryStdoutFormat) -> QueryOutputFormat {
    if quiet {
        QueryOutputFormat::Quiet
    } else {
        format.into()
    }
}

pub(crate) fn read_stdin_string() -> io::Result<String> {
    let mut buf = String::new();
    io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}

/// Read query document text from a positional arg (file path or inline) or stdin.
pub(crate) fn read_query_payload(query: Option<&str>) -> io::Result<(String, Option<String>)> {
    let Some(arg) = query else {
        return read_stdin_string().map(|s| (s, None));
    };
    if arg == "-" {
        return read_stdin_string().map(|s| (s, None));
    }
    let path = Path::new(arg);
    if path.is_file() {
        let text = fs::read_to_string(path)?;
        Ok((text, Some(arg.to_owned())))
    } else {
        Ok((arg.to_owned(), None))
    }
}

/// JSON vs TOML for a payload read by [`read_query_payload`].
pub(crate) fn query_input_format(path_hint: Option<&str>, text: &str) -> QueryInputFormat {
    detect_query_input_format(path_hint, text)
}
