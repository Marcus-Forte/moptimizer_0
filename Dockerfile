FROM  ubuntu:latest

RUN apt-get update && apt-get install -y build-essential cmake libtbb-dev libeigen3-dev libgtest-dev libgmock-dev

COPY . /src/

RUN mkdir /src/build && \
    cd /src/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON && \
    make -j4

RUN cd /src/build && ctest --output-on-failure
