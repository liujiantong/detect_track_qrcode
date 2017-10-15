# detect_track_qrcode



## install dependencies

### [opencv-3.3.0 + contrib](https://opencv.org/)

### [spdlog](https://github.com/gabime/spdlog)

### [cpputest](http://cpputest.github.io/)

### zbar
```
brew install zbar
LDFLAGS=-L/usr/local/lib/ CPATH=/usr/local/include/ pip install git+https://github.com/npinchot/zbar.git
```

## build
```
mkdir build
cd build/
cmake .. OR
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```
