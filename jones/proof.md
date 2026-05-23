# Proof for Greedy Two-Layer Neural Network Approximation

Let $\sigma$ be a uniformly bounded function. Suppose $f(x)$ is a function on $X$ and can be written as
$$f(x) = \mathbb{E}_{(w, b)\sim \Theta}[ \sigma(w\cdot x + b) ]$$
where the expectation is taken on a joint probability distribution $\Theta$ for $(w, b)$. 

But we assume the distribution $\Theta$ is NOT given to us (though it exists). We consider the following greedy algorithm to find the approximation of $n$ terms.

> **Algorithm Iteration**
> 
> Define $h_0(x) = 0$. The iterations are calculated as follows:
> $$(\alpha_k, w_k, b_k) := \arg\min_{\alpha\in [0, 1], (w, b)\in\mathbb{R}^{d+1}} \|f(x) - ((1-\alpha) h_k + \alpha \sigma(w\cdot x + b)) \|_{L^2(X)}$$
> and
> $$h_{k+1} = (1-\alpha_k) h_k + \alpha_k \sigma(w_k\cdot x + b_k).$$

---

### Questions

1. At the iteration $k\ge 0$, prove that 
   $$\mathbb{E}_{(w, b)\sim\Theta} \langle h_k - f, \sigma(w\cdot x + b) - f \rangle = 0.$$
   And show that there exists $(w', b')\in\mathbb{R}^{d+1}$ such that
   $$\langle h_{k} - f, \sigma(w'\cdot x + b') - f \rangle \le 0.$$

2. Let $e_{k+1} = \|f - h_{k+1}\|_{L^2(X)}$. Let $(w',b')$ be the parameters from the previous question for iteration $k$. Show that 
   $$e_{k+1}^2 \le \inf_{\alpha\in[0, 1]} \left( (1-\alpha)^2 e_{k}^2 + \alpha^2 \|f - \sigma(w'\cdot x + b') \|^2 \right)$$

3. Let
   $$M^2 := \sup_{(w, b)\in\mathbb{R}^{d+1}}\|f - \sigma(w\cdot x + b) \|^2.$$
   Prove that 
   $$\frac{1}{e_{k+1}^2} \ge \frac{1}{M^2} + \frac{1}{e_{k}^2}.$$

4. Show that after $n$ iterations of the algorithm, the resulting function $h_n$ satisfies
   $$\|h_n - f\|_{L^2} \le \frac{M}{\sqrt{n}}.$$
