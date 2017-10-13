# detect_track_qrcode



## install dependencies

## zbar
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
