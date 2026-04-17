"""test subcommand: run each module's free test() function."""

import sys
import argparse
import traceback

import claudex.rrf as rrf
import claudex.search as search
import claudex.con_com as con_com


MODULES = [con_com, rrf, search]


def cmd_test(args: argparse.Namespace):
    failed = 0

    for m in MODULES:
        name = m.__name__

        try:
            m.test()
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

            continue

        print(f"ok   {name}")

    if failed:
        sys.exit(1)
