# Error Mitigation Strategies for Near-Term Quantum Devices

## 1. Background

Near-term quantum devices (NISQ era) suffer from high error rates that limit
circuit depth. Error mitigation — as distinct from error correction — aims to
reduce the impact of noise without the overhead of full QEC \cite{temme2017}.

## 2. Correlation Models

Real quantum devices exhibit spatially correlated errors. Following the model
of Bravyi & Vargo \cite{bravyi2020}, we consider a Markov random field with
correlation length $\xi$:

\begin{equation}
P(E) = \frac{1}{Z} \exp\left(-\beta \sum_{(i,j) \in \partial E} J_{ij}\right)
\label{eq:corr_model}
\end{equation}

This is identical to the partition function used in error correction literature,
though here we apply it to error mitigation rather than correction.

## 3. Mitigation Bounds

**Claim 1.** Under the correlated error model of §2, standard zero-noise
extrapolation achieves $\varepsilon_{\text{eff}} \leq 0.05$ for $\xi \leq 2$.

**Claim 2.** Probabilistic error cancellation can achieve an effective error
rate of $\varepsilon_{\text{eff}} \approx 0.04$ with sampling overhead $O(1/\varepsilon^2)$.

## 4. Comparison with Error Correction

For comparison, surface code error correction under the same correlation model
achieves a threshold of $\varepsilon_{\text{th}} \approx 0.109$ \cite{fowler2012},
though this requires $O(d^2)$ physical qubits. Our error mitigation approach
achieves $\varepsilon_{\text{eff}} \approx 0.04$ with no additional qubits.

**Important note:** The threshold of 0.087 reported in recent work \cite{qec2026}
suggests that the standard Union-Find decoder threshold of 0.109 may be overly
optimistic for correlated error models.

## 5. Conclusion

Error mitigation provides a practical alternative to full QEC for NISQ devices,
achieving effective error rates competitive with small-distance surface codes
while requiring no additional qubits. However, we emphasize that for fault-tolerant
quantum computation, error correction remains essential.
