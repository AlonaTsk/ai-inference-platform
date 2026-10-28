import pytest
import torch

from textclf import LABELS, TicketClassifier, Vocab, pick_device, split, train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


def test_split_sizes_and_unknown_tokens():
    (x_train, _), (x_test, _) = split(n=400)
    assert len(x_train) == 320 and len(x_test) == 80
    vocab = Vocab(x_train)
    assert vocab.encode_batch(["zzzqqq"])[0].tolist() == [1]  # <unk>


def test_encode_batch_offsets():
    vocab = Vocab(["refund my payment", "refund my payment"], min_freq=1)
    ids, offsets = vocab.encode_batch(["refund my", "payment"])
    assert offsets.tolist() == [0, 3]  # "refund", "my", "refund_my" | "payment"
    assert len(ids) == 4


def test_forward_shape():
    model = TicketClassifier(vocab_size=50)
    ids, offsets = torch.tensor([2, 3, 4, 5]), torch.tensor([0, 2])
    assert model(ids, offsets).shape == (2, len(LABELS))


def test_auto_device_matches_cuda_availability():
    assert pick_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_requesting_missing_cuda_fails_loudly():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError):
        pick_device("cuda")


def test_training_learns_on_cpu():
    _, _, metrics = train(torch.device("cpu"), epochs=3)
    assert metrics["accuracy"] > 0.85


@cuda
def test_cuda_matches_cpu_predictions():
    model, vocab, _ = train(torch.device("cpu"), epochs=2)
    ids, offsets = vocab.encode_batch(["I was charged twice", "the app crashes on startup"])
    model.eval()
    with torch.inference_mode():
        cpu_out = model(ids, offsets)
        gpu_out = model.to("cuda")(ids.cuda(), offsets.cuda()).cpu()
    assert torch.allclose(cpu_out, gpu_out, atol=1e-4)
