import sys

def run(args):
  assert len(args) == 0
  print """
LIBTBX_OPT_RESOURCES=False|True
  If True, use fast math libraries from opt_resources directory if
  available.

LIBTBX_DISABLE_TRACEBACKLIMIT
  If set, Sorry and Usage exceptions are shown with the full traceback.

LIBTBX_VALGRIND
  Run "libtbx.valgrind python" for more information.

LIBTBX_PRINT_TRACE
  If set, print trace of all Python code executed.
  This can lead to very large output.

LIBTBX_NATIVE_TAR
  Inspected by libtbx.bundle_as_selfx to find alternative tar command.
  Example: setenv LIBTBX_NATIVE_TAR $HOME/bin/tar

LIBTBX_FULL_TESTING
  If set, forces libtbx.env.full_testing = True.

LIBTBX_DEBUG_LOG
  If set, enables libtbx.introspection.method_debug_log.
  See method_debug_log documentation for details.
"""

if (__name__ == "__main__"):
  run(sys.argv[1:])
