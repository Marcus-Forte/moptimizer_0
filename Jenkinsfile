pipeline {
    agent any

    stages {
        stage ('Dependencies') {
            steps {
               // sh 'sudo apt install -y libeigen3-dev libpcl-dev libgtest-dev'
            }
        }
        stage('Build') {
            steps {
                sh 'mkdir -p build'
                dir('build') {
                    sh 'cmake ..'
                    sh 'make'
                }
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                dir('build') {
                    sh 'ctest'
                }
            }
        }
        stage ('Profile') {
            steps {
                dir('build') {
                    sh 'valgrind --tool=callgrind bin/slam '
                }
            }
        }
    }
}

