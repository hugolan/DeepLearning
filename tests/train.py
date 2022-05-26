# for terminal
# from tqdm import tqdm
# for notebooks
import tqdm.notebook as tqdm
from model import *

train_dir = "train_data.pkl"
val_dir = "val_data.pkl"


def compute_psnr(denoised, target, max_range=1.0):
    assert denoised.shape == target.shape and denoised.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(
        ((denoised - target) ** 2).mean((1, 2, 3))).mean()


def train_model(load_model=False, save_model=False, model=None, optimizer=None, loss_fn=None,
                batch_size=100, num_epochs=1):
    model = Model(model=model)

    train_input0, train_input1 = torch.load(train_dir)
    val_input, val_target = torch.load(val_dir)

    train_input0 = train_input0.float() / 255.0
    train_input1 = train_input1.float() / 255.0
    val_input = val_input.float() / 255.0
    val_target = val_target.float() / 255.0

    output_psnr_before = compute_psnr(val_input, val_target)

    if load_model:
        model.load_pretrained_model()
    else:
        model.train(train_input0, train_input1, optimizer=optimizer, loss_fn=loss_fn, num_epochs=num_epochs,
                    batch_size=batch_size)
        if save_model:
            model.save_trained_model()

    mini_batch_size = 100
    model_outputs = []
    for b in tqdm(range(0, val_input.size(0), mini_batch_size)):
        output = model.predict(val_input.narrow(0, b, mini_batch_size))
        model_outputs.append(output.cpu())
    model_outputs = torch.cat(model_outputs, dim=0)

    output_psnr_after = compute_psnr(model_outputs, val_target)
    print(f"[PSNR : {output_psnr_after:.2f} dB], PSNR before training = {output_psnr_before:.2f} dB")
    return model_outputs
