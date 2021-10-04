import torch
import scipy
import matplotlib.pyplot as plt

v = 30
t2 = 1
chi = torch.distributions.chi2.Chi2(v)
data = v/chi.sample_n(5000)

plt.hist(data.unsqueeze(0), color = 'blue', edgecolor = 'black',
         bins = 100)


# Add labels
plt.title('Histogram of Arrival Delays')
plt.xlabel('Delay (min)')
plt.ylabel('Flights')
plt.show()