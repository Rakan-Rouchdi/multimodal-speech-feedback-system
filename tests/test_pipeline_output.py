from app.pipeline.runner import run_pipeline


def test_pipeline_output_keys_multimodal():
    # Use a small local audio file you have (not committed) - test will be skipped if missing
    file_path = "data/harvard.wav"
    try:
        result = run_pipeline(file_path, "multimodal")
    except FileNotFoundError:
        return  # skip in environments without the file

    assert "meta" in result
    assert "scores" in result
    assert "debug" in result
    assert result["meta"]["pipeline_variant"] == "multimodal"
