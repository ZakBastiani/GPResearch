import torch
import matplotlib.pyplot as plt

v = 50
t2 = 1
chi = torch.distributions.chi2.Chi2(v)
# data = v/chi.sample((5000,))
gamma = torch.distributions.gamma.Gamma(v/2, v*t2/2)
data = 1/gamma.sample((5000,))
plt.hist(data.unsqueeze(0), color='blue', edgecolor='black',
         bins=100)


# Add labels
plt.title('Histogram of Arrival Delays')
plt.xlabel('Delay (min)')
plt.ylabel('Flights')
plt.show()
