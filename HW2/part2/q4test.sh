echo "[View 1] 4 threads:"
./mandelbrot -t 4

echo "[View 1], 8 threads:"
./mandelbrot -t 8

echo "[View 2], 4 threads:"
./mandelbrot -v 2 -t 4

echo "[View 2], 8 threads:"
./mandelbrot -v 2 -t 8