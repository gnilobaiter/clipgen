from src.core.editor import WhisperProgressStream


def test_whisper_progress_stream_throttled_output():
    logs = []
    stream = WhisperProgressStream(logger=logs.append)

    # 1st segment -> logs immediately
    stream.write("[00:00.000 --> 00:05.000] Hello world\n")
    assert len(logs) == 1
    assert "00:00.000" in logs[0]
    assert "Hello world" in logs[0]

    # 2nd to 10th segments -> throttled
    for i in range(1, 10):
        stream.write(f"[00:0{i}.000 --> 00:0{i+1}.000] Segment {i}\n")
    assert len(logs) == 1

    # 11th segment -> logs
    stream.write("[00:10.000 --> 00:11.000] Tenth segment\n")
    assert len(logs) == 2
    assert "00:10.000" in logs[1]
    assert "Tenth segment" in logs[1]


def test_whisper_progress_stream_no_bracket():
    logs = []
    stream = WhisperProgressStream(logger=logs.append)
    stream.write("00:00.000 --> 00:05.000 Unbracketed timestamp\n")
    assert len(logs) == 1
    assert "00:00.000" in logs[0]


def test_whisper_progress_stream_empty_or_non_timestamp():
    logs = []
    stream = WhisperProgressStream(logger=logs.append)
    stream.write("\n")
    stream.write("Random debug line from whisper\n")
    assert len(logs) == 0


def test_whisper_progress_stream_flush():
    stream = WhisperProgressStream(logger=lambda msg: None)
    # flush should execute without error
    stream.flush()


def test_progress_stream_aborts_transcription_when_cancelled():
    import pytest

    from src.core.editor import TranscriptionCancelled

    state = {"cancel": False}
    stream = WhisperProgressStream(logger=None, is_cancelled=lambda: state["cancel"])
    stream.write("[00:00.000 --> 00:05.000] fine\n")
    state["cancel"] = True
    with pytest.raises(TranscriptionCancelled):
        stream.write("[00:05.000 --> 00:10.000] too late\n")
    assert WhisperProgressStream(logger=None).write("x") == 1  # no callback: never raises
