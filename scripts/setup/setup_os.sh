#!/bin/bash

source /etc/os-release

export CUDAARCHS="89"

if [[ "$ID" == "arch" ]]; then

    sudo pacman -S python-protobuf python-flatbuffers python-pip python-setuptools python-virtualenv gcc-14 cuda ninja meson aria2 ccache cmake make ffmpeg python-gobject glib2 glib2-devel --noconfirm

elif [[ "$ID" == "fedora" ]]; then

    sudo dnf5 install python3-protobuf python3-flatbuffers python3-pip python3-setuptools python3-virtualenv gcc-14 cuda ninja-build meson aria2 ccache cmake make ffmpeg -y

elif [[ "$ID" == "rhel" || "$ID" == "centos" ]]; then

    sudo yum install python3-protobuf python3-flatbuffers python3-pip python3-setuptools python3-virtualenv gcc-14 cuda ninja-build meson aria2 ccache cmake make ffmpeg -y

elif [[ "$ID" == "ubuntu" || "$ID" == "debian" ]]; then

    sudo apt-get update
    sudo apt-get install python3-protobuf python3-flatbuffers python3-pip python3-setuptools python3-virtualenv gcc-14 cuda ninja-build meson aria2 ccache cmake make ffmpeg -y
    

else
    echo "Unsupported or unknown OS: $ID"
fi

