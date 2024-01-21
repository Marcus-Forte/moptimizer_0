
#ifndef MOPTIMIZER_EXPORT_H
#define MOPTIMIZER_EXPORT_H

#ifdef MOPTIMIZER_STATIC_DEFINE
#  define MOPTIMIZER_EXPORT
#  define MOPTIMIZER_NO_EXPORT
#else
#  ifndef MOPTIMIZER_EXPORT
#    ifdef moptimizer_EXPORTS
        /* We are building this library */
#      define MOPTIMIZER_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define MOPTIMIZER_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef MOPTIMIZER_NO_EXPORT
#    define MOPTIMIZER_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef MOPTIMIZER_DEPRECATED
#  define MOPTIMIZER_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef MOPTIMIZER_DEPRECATED_EXPORT
#  define MOPTIMIZER_DEPRECATED_EXPORT MOPTIMIZER_EXPORT MOPTIMIZER_DEPRECATED
#endif

#ifndef MOPTIMIZER_DEPRECATED_NO_EXPORT
#  define MOPTIMIZER_DEPRECATED_NO_EXPORT MOPTIMIZER_NO_EXPORT MOPTIMIZER_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef MOPTIMIZER_NO_DEPRECATED
#    define MOPTIMIZER_NO_DEPRECATED
#  endif
#endif

#endif /* MOPTIMIZER_EXPORT_H */
