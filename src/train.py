import math
import time

from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm

from src.utils import MeanAggregate
from src.generation import generate


def train(loader, model, criterion, optimizer, log_interval=100, epoch=1, grad_clip=5.,
          gen_interval=None, gen_max_length=None, num_gens=3):
    model.train()
    loss = MeanAggregate()
    runtime = MeanAggregate()
    ppl = MeanAggregate()
    speed = MeanAggregate()

    for k, (inputs, targets, seq_lens) in enumerate(loader):
        batch_start_time = time.time()
        inputs, targets = Variable(inputs), Variable(targets)
        init_states = model.init_states(inputs.size(1))
        outputs, _ = model(inputs, init_states, seq_lens=seq_lens)
        batch_loss = criterion(outputs, targets)
        batch_ppl = math.exp(batch_loss.data[0])
        optimizer.zero_grad()
        batch_loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        batch_runtime = time.time() - batch_start_time

        loss.update(batch_loss.data[0])
        runtime.update(batch_runtime)
        ppl.update(batch_ppl)
        speed.update(loader.batch_size/batch_runtime)

        if (k + 1) % log_interval == 0:
            print(f'Epoch {epoch} [{k+1}/{len(loader)}]:', end=' ')
            print(f'loss {loss.mean:.4f} | ppl {ppl.mean:.4f}', end=' | ')
            print(f'{runtime.mean*1000:.2f}ms | {speed.mean:.2f} samples/s')
        if gen_interval is not None and (k + 1) % gen_interval == 0:
            print('Generating samples...')
            for j in range(num_gens):
                print(f'Epoch {epoch} [{k+1}/{len(loader)}] gen {j}:', end=' ')
                res = generate(model, loader.dataset, max_length=gen_max_length)
                print(''.join(res[1:]))

    print(f'Epoch {epoch} done in {runtime.total:.2f}s')
    return loss.mean, ppl.mean


def evaluate(loader, model, criterion):
    model.eval()
    loss = MeanAggregate()
    ppl = MeanAggregate()
    for inputs, targets, seq_lens in loader:
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        init_states = model.init_states(inputs.size(1))
        outputs, _ = model(inputs, init_states, seq_lens=seq_lens)
        batch_loss = criterion(outputs, targets)
        batch_ppl = math.exp(batch_loss.data[0])
        loss.update(batch_loss.data[0])
        ppl.update(batch_ppl)
    return loss.mean, ppl.mean


def cross_entropy_loss(outputs, targets):
    outputs, targets = outputs.view(-1, outputs.size(2)), targets.view(-1)
    index = Variable((targets.data != -1).nonzero().squeeze())
    return F.cross_entropy(outputs.index_select(0, index), targets.index_select(0, index))
