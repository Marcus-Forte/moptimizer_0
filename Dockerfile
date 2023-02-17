FROM  marcusforte/nomad-base

COPY . /src/

RUN mkdir /src/build && \
    cd /src/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON && \
    make -j4

RUN cd /src/build && ctest --output-on-failure