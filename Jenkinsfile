pipeline {
    agent any

    stages {
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
    }
}

