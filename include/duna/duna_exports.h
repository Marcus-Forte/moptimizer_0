
#ifndef DUNA_OPTIMIZER_EXPORT_H
#define DUNA_OPTIMIZER_EXPORT_H

#ifdef DUNA_OPTIMIZER_STATIC_DEFINE
#  define DUNA_OPTIMIZER_EXPORT
#  define DUNA_OPTIMIZER_NO_EXPORT
#else
#  ifndef DUNA_OPTIMIZER_EXPORT
#    ifdef duna_optimizer_EXPORTS
        /* We are building this library */
#      define DUNA_OPTIMIZER_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define DUNA_OPTIMIZER_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef DUNA_OPTIMIZER_NO_EXPORT
#    define DUNA_OPTIMIZER_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef DUNA_OPTIMIZER_DEPRECATED
#  define DUNA_OPTIMIZER_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef DUNA_OPTIMIZER_DEPRECATED_EXPORT
#  define DUNA_OPTIMIZER_DEPRECATED_EXPORT DUNA_OPTIMIZER_EXPORT DUNA_OPTIMIZER_DEPRECATED
#endif

#ifndef DUNA_OPTIMIZER_DEPRECATED_NO_EXPORT
#  define DUNA_OPTIMIZER_DEPRECATED_NO_EXPORT DUNA_OPTIMIZER_NO_EXPORT DUNA_OPTIMIZER_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef DUNA_OPTIMIZER_NO_DEPRECATED
#    define DUNA_OPTIMIZER_NO_DEPRECATED
#  endif
#endif

#endif /* DUNA_OPTIMIZER_EXPORT_H */
