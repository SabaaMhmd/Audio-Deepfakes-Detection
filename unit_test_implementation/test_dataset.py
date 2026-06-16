import sys
sys.path.append("software_implementation/aasist")

from data_utils import Dataset_ASVspoof2019_train

def test_dataset_length():

    dummy_files = [
        "audio1.wav",
        "audio2.wav",
        "audio3.wav"
    ]

    labels = {
        "audio1.wav": 0,
        "audio2.wav": 1,
        "audio3.wav": 0
    }

    dataset = Dataset_ASVspoof2019_train(
        dummy_files,
        labels,
        None
    )

    assert len(dataset) == 3
