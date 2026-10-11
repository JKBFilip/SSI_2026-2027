import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
t = np.arange(1.0, 2.01, 0.01)
x1 = t * np.sin(t * 2 * np.pi) - 0.1 + 0.2 * np.random.rand(len(t))
x2 = t * np.cos(t * 2 * np.pi) - 0.1 + 0.2 * np.random.rand(len(t))
probki = np.column_stack((x1, x2))

def rysuj_kmeans(dane, k, iteracje, tytul, nazwa_pliku, tylko_x1=False):
    np.random.seed(42)
    srodki = dane[np.random.choice(len(dane), k, replace=False)].copy()
    
    for _ in range(iteracje):
        if tylko_x1:
            dystanse = np.abs(dane[:, 0:1] - srodki[:, 0:1].T)
        else:
            dystanse = np.linalg.norm(dane[:, None] - srodki, axis=2)
        
        przypisania = np.argmin(dystanse, axis=1)
        
        for i in range(k):
            if np.any(przypisania == i):
                srodki[i] = dane[przypisania == i].mean(axis=0)
    
    plt.figure(figsize=(8, 6))
    kolory = ['red', 'green', 'blue', 'purple']
    
    for i in range(k):
        punkty_grupy = dane[przypisania == i]
        plt.scatter(punkty_grupy[:, 0], punkty_grupy[:, 1], c=kolory[i], label=f'Grupa {i+1}', marker='s', edgecolors='black', alpha=0.6)
        plt.scatter(srodki[i, 0], srodki[i, 1], c=kolory[i], marker='*', s=300, edgecolors='black')
    
    plt.title(tytul)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(nazwa_pliku)
    plt.close()

rysuj_kmeans(probki, k=3, iteracje=4, tytul="K-Średnie (Euklidesowa) - 4 iteracje", nazwa_pliku="wykres_1_4iters.png")
rysuj_kmeans(probki, k=3, iteracje=10, tytul="K-Średnie (Euklidesowa) - 10 iteracji", nazwa_pliku="wykres_1_10iters.png")
rysuj_kmeans(probki, k=4, iteracje=10, tytul="K-Średnie (Tylko X1) - 10 iteracji", nazwa_pliku="wykres_2_10iters.png", tylko_x1=True)

