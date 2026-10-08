"""
[study] Environment check: evaluate akamaster's pretrained weights on the CIFAR-10 test set (10k images)
and compare with the README. If these match, our data pipeline / model code are correct.

    python eval_pretrained.py --data-dir ./data --archs resnet20 resnet56
"""
import argparse, csv, glob, os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import resnet

README_ERR = {'resnet20': 8.27, 'resnet32': 7.37, 'resnet44': 6.90, 'resnet56': 6.61, 'resnet110': 6.32}
PAPER_ERR = {'resnet20': 8.75, 'resnet32': 7.51, 'resnet44': 7.17, 'resnet56': 6.97, 'resnet110': 6.43}

p = argparse.ArgumentParser()
p.add_argument('--archs', nargs='+', default=['resnet20', 'resnet56'])
p.add_argument('--data-dir', default='./data')
p.add_argument('--out', default='results/pretrained_eval.csv')
a = p.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # same as training
test_set = datasets.CIFAR10(a.data_dir, train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False, num_workers=2)

rows = []
for arch in a.archs:
    path = glob.glob(os.path.join(os.path.dirname(__file__) or '.', 'pretrained_models', f'{arch}-*.th'))[0]
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in ck['state_dict'].items()}   # saved via DataParallel
    model = resnet.__dict__[arch]().to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    err = 100 * (1 - correct / len(test_set))
    rows.append([arch, os.path.basename(path), f'{err:.2f}', README_ERR.get(arch), PAPER_ERR.get(arch)])
    print(f'{arch:10s} measured {err:5.2f}% | README {README_ERR.get(arch)}% | paper {PAPER_ERR.get(arch)}%')

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['arch', 'file', 'measured_test_err', 'readme_err', 'paper_err']); w.writerows(rows)
print('saved', a.out)
