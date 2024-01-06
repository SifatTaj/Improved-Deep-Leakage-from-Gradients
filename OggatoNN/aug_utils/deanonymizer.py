import torch
import torchvision
from torchvision.transforms import transforms

from aug_models.anon_densenet import OriginalDenseNet121


def deanonymize(aug_net, original_net, device):
    trained_weights = {}

    for name, parameter in aug_net.named_parameters():
        name = name.replace('module.', '')
        trained_weights[name] = parameter.data.cpu().numpy()

    for name, parameter in original_net.named_parameters():
        parameter.data = torch.from_numpy(trained_weights[name]).to(device)

    return original_net


def check_accuracy(model, dataloader):
    total_sample = 0
    correct_sample = 0

    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores, x = model(x)
            _, predictions = scores.max(1)
            correct_sample += (y == predictions).sum()
            total_sample += predictions.size(0)

    model.train()

    print(
        f"out of total sample : {total_sample}  correct sample : {correct_sample} accuracy : {float(correct_sample / total_sample) * 100:.2f}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = torch.load('../trained_models/model_2022_08_09-12:45:30_PM.pt')
    original_net = OriginalDenseNet121(num_classes=10, grayscale=False)
    deanon_net = deanonymize(net, original_net, device)

    deanon_net.to(device)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    check_accuracy(deanon_net, testloader)
