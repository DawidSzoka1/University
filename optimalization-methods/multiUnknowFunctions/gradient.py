from myFunction import *


def calculate(x0, y0, epsilon, iterations=100, function=function, pochx=functionPochx, pochy=functionPochy,
              poch2x=functionPoch2x, poch2y=functionPoch2y, pochxy=functionPochxy, check_e=True):
    xk, yk = x0, y0
    trajectory = [(xk, yk)]
    for i in range(iterations):
        grad = gradient(xk, yk, pochx=pochx, pochy=pochy)
        hass = hassego(xk, yk, pochxx=poch2x, pochyy=poch2y, pochxy=pochxy)
        alpha_k = calculate_alpha(grad, hass)
        xk2, yk2 = np.array([xk, yk]) - alpha_k * grad
        trajectory.append((xk2, yk2))
        grad_next = gradient(xk2, yk2, pochx=pochx, pochy=pochy)
        if check_e:
            if np.linalg.norm(grad_next) <= epsilon or (abs(xk2 - xk) <= epsilon and abs(yk2 - yk) <= epsilon):
                return xk2, yk2, i + 1, np.array(trajectory)
        xk, yk = xk2, yk2
    return xk, yk, iterations, np.array(trajectory)


if __name__ == "__main__":
    test = calculate(2, 2, 0.01, 100, check_e=False)
    test2 = calculate(10, 12, 0.01, 100, pochxy=testFunctionPochxy, pochx=testFunctionPochx, pochy=testFunctionPochy,
                      poch2x=testFunctionPochxx, poch2y=testFUnctionPochyy)
    plot_trajectory(example_function, test2[3])
    plot_trajectory(function, test[3])
    print(test2)
