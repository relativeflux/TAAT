import os
import matplotlib.pyplot as plt


def plot_olaf_api_result(result):
    labels = []
    cnts = []
    total_matches = 0
    for match in result:
        cnt = match['matchCount']
        start = f'{match["queryStart"]:.2f}'
        stop = f'{match["queryStop"]:.2f}'
        path = str(match['path'], encoding='utf-8')
        path = os.path.basename(path)
        labels.append(f'{path}_[start={start},stop={stop}]')
        total_matches += cnt
        cnts.append(cnt)
    sizes = [f'{100 * (cnt / total_matches):.2f}' for cnt in cnts]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()
    
