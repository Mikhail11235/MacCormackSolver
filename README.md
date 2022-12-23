# MacCormackSolver
MacCormack solver for Burgers equation

Burgers equation:
$$\frac{\partial{u}}{\partial{t}} + u_x \frac{\partial{u}}{\partial{x}} = \frac{1}{Re}\frac{\partial^2{u}}{\partial{x}^2}$$
where 
$$u(-50, 0) = u(-50, t) = -1$$
$$u(50, 0) = u(50, t) = 1$$
$$u(x, 0) = 0, x \in (-50, 50)$$

MacCormack method:
$$u_j^{\overline{n + 1}} = u_j^n - \frac{\Delta t}{\Delta x} (F_{j+1}^n - F_{j}^n) + r (u_{j+1}^n - 2 u_{j}^n + u_{j-1}^n)$$

$$u_j^{n + 1} = \frac{1}{2}\Bigl( u_j^n + u_j^{\overline{n + 1}} - \frac{\Delta t}{\Delta x} (F^{\overline{n + 1}}_j - F^{\overline{n + 1}}_{j-1})+ r (u_{j+1}^{\overline{n + 1}} - 2 u_{j}^{\overline{n + 1}} + u_{j-1}^{\overline{n + 1}})\Bigr)$$

where $r = \frac{\Delta t}{(\Delta x)^2 * Re}$ 

<hr>

![maccormack-all](https://user-images.githubusercontent.com/59762084/209370168-04848ae6-46fa-4c03-b75e-852e625ecc3b.png)
