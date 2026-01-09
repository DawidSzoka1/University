from myFunction import *


def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=dfdx, pochy=dfdy,
              poch2x=dfdxdx, poch2y=dfdydy, pochxy=dfdxdy, check_e=True):
    xk, yk = x0, y0
    trajectory = [(xk, yk)]
    for i in range(iterations):
        grad = gradientF(xk, yk, fun=function)
        hass = hassego(xk, yk, func=function, pochxx=poch2x, pochyy=poch2y, pochxy=pochxy)
        alpha_k = calculate_alpha(grad, hass)
        xk2, yk2 = np.array([xk, yk]) - alpha_k * grad
        trajectory.append((xk2, yk2))
        grad_next = gradientF(xk2, yk2, fun=function)
        if check_e:
            if np.linalg.norm(grad_next) <= epsilon or (abs(xk2 - xk) <= epsilon and abs(yk2 - yk) <= epsilon):
                return xk2, yk2, i + 1, np.array(trajectory)
        xk, yk = xk2, yk2
    return xk, yk, iterations, np.array(trajectory)


if __name__ == "__main__":
    test = calculate(2, 2, 0.01, 100, check_e=False)
    tes2 = calculate(10, 12, 0.01, 100, function=example_function)
    plot_trajectory(function, test[3])
