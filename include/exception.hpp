#pragma once
#include <exception>
#include <string>


namespace duna {

class Exception : public std::exception {
public:
    Exception(const std::string& msg) noexcept : m_msg(msg){}
    virtual ~Exception() noexcept {}


protected:

    virtual const char* what() const noexcept override{
        return m_msg.c_str();
    }
    std::string m_msg;

};

}