export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export GI_TYPELIB_PATH=/usr/local/lib/girepository-1.0:$GI_TYPELIB_PATH


source ./scripts/setup/gcc_switcher.sh

cd ./thirdparty/gstreamer
 
sudo ninja -C builddir uninstall

rm -rf builddir

meson setup builddir \
  --prefix=/usr/local \
  --buildtype=release \
  -Dgpl=enabled

ninja -C builddir
sudo ninja -C builddir install

source ../../.venv/bin/activate

cd ./subprojects/gst-python

rm -rf builddir

meson setup builddir \
  --prefix=/usr \
  --buildtype=release
 
ninja -C builddir
meson install -C builddir