# Use an official Python runtime as a parent image
FROM python:3.8

# Install necessary packages for PyQt5, Qt XCB platform, Xvfb, and other PyQt5 packages
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    xvfb \
    x11-xserver-utils \
    python3-pyqt5.qtopengl \
    python3-pyqt5.qtquick \
    python3-pyqt5.qtmultimedia \
    qmlscene \
    qml-module-qtqml* \
    qml-module-qtquick* \
    qml-module-qt-websockets \
    qml-module-qt3d \
    qml-module-qtaudioengine \
    qml-module-qtcharts \
    qml-module-qtdatavisualization \
    qml-module-qtgraphicaleffects \
    qml-module-qtgstreamer \
    qml-module-qtlocation \
    qml-module-qtmultimedia \
    qml-module-qtpositioning \
    libqt5multimedia5-plugins \
    gstreamer1.0-libav \
    gstreamer1.0-alsa \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the HiTIPS directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in HiTIPS/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to start Xvfb and the application
ENTRYPOINT ["xvfb-run", "-s", "-screen 0 1024x768x24", "python", "-m", "hitips.HiTIPS"]
