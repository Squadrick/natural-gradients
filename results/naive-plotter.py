import matplotlib.pyplot as plt

def get_results_table(filename):
    f = open(filename, 'r')
    data = []
    for line in f:
        data.append(float(line[:-2]))
    return data


cov = get_results_table('natgrad-naive-cov-results')
nat = get_results_table('natgrad-naive-nat-results')

plt.plot(cov)
plt.plot(nat)

plt.savefig('naive-results.png')
