import torch
from torch.distributions import multinomial
from tools.utils import set_figsize
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimate = cum_counts / cum_counts.sum(dim=1, keepdim=True)

set_figsize((6, 4.5))
for i in range(6):
    plt.plot(estimate[:, i].numpy(),
             label=("p(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.xlabel('Groups of experiments')
plt.ylabel('Estimated probability')
plt.legend()
plt.show()