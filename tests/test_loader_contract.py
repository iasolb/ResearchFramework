"""What `ResearchHandler._load` and the loader functions actually promise.

This file replaces four tests deleted from test_handler.py on 2026-08-28 that
asserted `handler.data is None` on failure. That is a dead contract: `_load`
returns an EMPTY DataFrame and prints why. The behaviours were worth keeping,
so they are re-asserted here against the real contract.

TWO BUGS THIS PINS DOWN, both fixed the same day and both silent:

- a `str` path was rejected outright, while the class docstring documented
  `ResearchHandler("data.csv")`.
- the path branch computed its result and then fell through to the DataFrame
  check, landed in the else, and overwrote the loaded data with an empty frame.
  Every file load read the file and threw the result away.

AN OPEN DESIGN QUESTION, deliberately not asserted either way: operations on a
handler whose load failed (so `data` is an empty frame) raise KeyError rather
than warning, because the column being referenced is not there. Four deleted
tests asserted they should warn instead. Whether to guard the empty-data path
is a real decision; until it is made, a test asserting either behaviour would
be locking in an accident.
"""

import pickle

import pandas as pd
import pytest

from research_framework.rh import (
    ResearchHandler,
    csv_loader,
    json_loader,
    parquet_loader,
    pickle_loader,
    txt_loader,
)


def identity(df):
    return df


@pytest.fixture
def frame():
    return pd.DataFrame({"age": [25, 40, 35], "income": [1.0, 2.0, 3.0]})


@pytest.fixture
def csv_path(tmp_path, frame):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


# ── the accepted source types ─────────────────────────────────────────────


def test_a_str_path_loads(csv_path, frame):
    """The documented call. Rejected outright before 2026-08-28."""
    handler = ResearchHandler(str(csv_path), identity)
    assert len(handler.data) == len(frame)
    assert list(handler.data.columns) == list(frame.columns)


def test_a_path_object_loads(csv_path, frame):
    """Accepted before, but its result was discarded by the fall-through."""
    assert len(ResearchHandler(csv_path, identity).data) == len(frame)


def test_a_dataframe_is_taken_as_given(frame):
    assert len(ResearchHandler(frame, identity).data) == len(frame)


def test_the_handler_function_is_applied(csv_path):
    handler = ResearchHandler(csv_path, lambda df: df[df["age"] > 30])
    assert len(handler.data) == 2


def test_no_handler_is_allowed(csv_path, frame):
    assert len(ResearchHandler(csv_path, None).data) == len(frame)


# ── the format is inferred, or given ──────────────────────────────────────


def test_the_format_is_inferred_from_the_extension(csv_path, frame):
    """So the documented single-argument call works with no data_format."""
    assert len(ResearchHandler(csv_path, identity).data) == len(frame)


def test_an_explicit_data_format_wins_over_the_extension(tmp_path, frame):
    """A tab-separated file named .dat still loads when told it is txt."""
    path = tmp_path / "data.dat"
    frame.to_csv(path, sep="\t", index=False)
    handler = ResearchHandler(path, identity, data_format="txt")
    assert list(handler.data.columns) == list(frame.columns)


def test_an_unsupported_format_is_named_and_the_options_listed(tmp_path, capsys):
    path = tmp_path / "data.weird"
    path.write_text("nothing", encoding="utf-8")
    handler = ResearchHandler(path, identity)
    out = capsys.readouterr().out
    assert handler.data.empty
    assert "weird" in out
    assert "csv" in out          # lists what IS supported


# ── failure returns an empty frame, and says why ──────────────────────────


def test_a_missing_file_gives_an_empty_frame_not_none(capsys):
    """Replaces the deleted test_bad_filepath, which asserted None."""
    handler = ResearchHandler("nonexistent.csv", identity)
    assert isinstance(handler.data, pd.DataFrame)
    assert handler.data.empty
    assert "nonexistent.csv" in capsys.readouterr().out


def test_a_raising_handler_gives_an_empty_frame_not_none(csv_path, capsys):
    """Replaces the deleted test_bad_cleaning_function, which asserted None."""

    def bad_clean(df):
        raise ValueError("intentional error")

    handler = ResearchHandler(csv_path, bad_clean)
    assert isinstance(handler.data, pd.DataFrame)
    assert handler.data.empty
    assert "handler function" in capsys.readouterr().out


def test_a_raising_handler_on_a_dataframe_source_also_degrades(frame, capsys):
    def bad_clean(df):
        raise ValueError("intentional error")

    handler = ResearchHandler(frame, bad_clean)
    assert handler.data.empty
    assert "handler function" in capsys.readouterr().out


def test_an_unusable_source_type_is_reported(capsys):
    handler = ResearchHandler(12345, identity)
    assert handler.data.empty
    assert "Invalid source type" in capsys.readouterr().out


# ── the individual loaders ────────────────────────────────────────────────


def test_csv_loader_round_trips(tmp_path, frame):
    path = tmp_path / "f.csv"
    frame.to_csv(path, index=False)
    pd.testing.assert_frame_equal(csv_loader(path), frame)


def test_txt_loader_reads_tab_separated(tmp_path, frame):
    path = tmp_path / "f.txt"
    frame.to_csv(path, sep="\t", index=False)
    pd.testing.assert_frame_equal(txt_loader(path), frame)


def test_json_loader_round_trips(tmp_path, frame):
    path = tmp_path / "f.json"
    frame.to_json(path, orient="records")
    assert list(json_loader(path).columns) == list(frame.columns)


def test_parquet_loader_round_trips(tmp_path, frame):
    path = tmp_path / "f.parquet"
    try:
        frame.to_parquet(path)
    except (ImportError, ValueError) as exc:      # no pyarrow/fastparquet
        pytest.skip(f"parquet engine unavailable: {exc}")
    pd.testing.assert_frame_equal(parquet_loader(path), frame)


def test_pickle_loader_passes_a_plain_frame_through(tmp_path, frame):
    path = tmp_path / "f.pkl"
    with open(path, "wb") as fh:
        pickle.dump(frame, fh)
    pd.testing.assert_frame_equal(pickle_loader(path), frame)


def test_pickle_loader_returns_a_single_frame_from_a_one_entry_dict(tmp_path, frame):
    path = tmp_path / "f.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"only": frame}, fh)
    pd.testing.assert_frame_equal(pickle_loader(path), frame)


def test_pickle_loader_gives_an_empty_frame_for_an_empty_dict(tmp_path):
    path = tmp_path / "f.pkl"
    with open(path, "wb") as fh:
        pickle.dump({}, fh)
    assert pickle_loader(path).empty


def test_pickle_loader_merges_on_a_shared_non_numeric_key(tmp_path):
    """The documented flattening: dict-of-frames outer-merged on shared
    non-numeric columns."""
    left = pd.DataFrame({"date": ["a", "b"], "x": [1, 2]})
    right = pd.DataFrame({"date": ["b", "c"], "y": [3, 4]})
    path = tmp_path / "f.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"l": left, "r": right}, fh)

    out = pickle_loader(path)
    assert set(out.columns) == {"date", "x", "y"}
    assert len(out) == 3                      # outer merge over a, b, c


def test_pickle_loader_stacks_with_a_series_label_when_nothing_is_mergeable(tmp_path):
    """No shared non-numeric column, so the frames are concatenated and
    labelled by their dict key instead of merged."""
    left = pd.DataFrame({"x": [1, 2]})
    right = pd.DataFrame({"x": [3, 4]})
    path = tmp_path / "f.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"first": left, "second": right}, fh)

    out = pickle_loader(path)
    assert "series" in out.columns
    assert set(out["series"]) == {"first", "second"}
    assert len(out) == 4
