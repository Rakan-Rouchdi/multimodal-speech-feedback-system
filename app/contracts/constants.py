PIPELINE_VARIANTS = ("speech_only", "text_only", "multimodal")

BANDS = [
    ("Needs improvement", 0, 39),
    ("Developing", 40, 69),
    ("Strong", 70, 84),
    ("Excellent", 85, 100),
]

# Filler words — includes CrisperWhisper disfluency markers ([UH], [UM])
FILLER_WORDS = {
    "um", "uh", "erm", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "right", "okay",
}

# CrisperWhisper outputs disfluencies in bracket notation
DISFLUENCY_MARKERS = {"[UH]", "[UM]"}
