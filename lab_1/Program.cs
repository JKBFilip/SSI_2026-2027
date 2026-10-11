using System;
using System.Collections.Generic;
using System.Linq;

namespace LabSSI
{
    class Program
    {
        static void Main(string[] args)
        {
            List<double[]> probki = GenerateSamples();

            Console.WriteLine("=== 1. RAPORT (K-Srednie, Euklidesowa, grupy: 3) ===");
            KMeans(probki, k: 3, maxIters: 10, reportIters: new[] { 4, 10 }, onlyX1: false);

            Console.WriteLine("\n=== 2. RAPORT (K-Srednie, tylko odleglosc X1, grupy: 4) ===");
            KMeans(probki, k: 4, maxIters: 10, reportIters: new[] { 10 }, onlyX1: true);
        }

        static List<double[]> GenerateSamples()
        {
            var probki = new List<double[]>();
            Random rnd = new Random(123);
            for (double t = 1.0; t <= 2.0001; t += 0.01)
            {
                double r1 = rnd.NextDouble();
                double r2 = rnd.NextDouble();
                double x1 = t * Math.Sin(t * 2 * Math.PI) - 0.1 + 0.2 * r1;
                double x2 = t * Math.Cos(t * 2 * Math.PI) - 0.1 + 0.2 * r2;
                probki.Add(new double[] { x1, x2 });
            }
            return probki;
        }

        static void KMeans(List<double[]> probki, int k, int maxIters, int[] reportIters, bool onlyX1)
        {
            Random rnd = new Random(42);
            List<double[]> centers = new List<double[]>();
            var shuffled = probki.OrderBy(x => rnd.Next()).ToList();
            for (int i = 0; i < k; i++) centers.Add((double[])shuffled[i].Clone());

            for (int iter = 1; iter <= maxIters; iter++)
            {
                var clusters = new List<List<double[]>>();
                for (int i = 0; i < k; i++) clusters.Add(new List<double[]>());

                foreach (var p in probki)
                {
                    int bestCenter = 0;
                    double bestDist = double.MaxValue;
                    for (int i = 0; i < k; i++)
                    {
                        double dist = onlyX1 ? Math.Abs(p[0] - centers[i][0]) : 
                            Math.Sqrt(Math.Pow(p[0] - centers[i][0], 2) + Math.Pow(p[1] - centers[i][1], 2));
                        
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestCenter = i;
                        }
                    }
                    clusters[bestCenter].Add(p);
                }

                for (int i = 0; i < k; i++)
                {
                    if (clusters[i].Count > 0)
                    {
                        centers[i][0] = clusters[i].Average(p => p[0]);
                        centers[i][1] = clusters[i].Average(p => p[1]);
                    }
                }

                if (reportIters.Contains(iter))
                {
                    Console.WriteLine($"\nWyniki po {iter} iteracjach :");
                    for (int i = 0; i < k; i++)
                    {
                        if (clusters[i].Count == 0) continue;
                        double minX1 = clusters[i].Min(p => p[0]);
                        double maxX1 = clusters[i].Max(p => p[0]);
                        double minX2 = clusters[i].Min(p => p[1]);
                        double maxX2 = clusters[i].Max(p => p[1]);
                        Console.WriteLine($"Grupa {i + 1} & ({centers[i][0]:F4}, {centers[i][1]:F4}) & {clusters[i].Count} & {minX1:F4} & {maxX1:F4} & {minX2:F4} & {maxX2:F4} \\\\");
                    }
                }
            }
        }
    }
}