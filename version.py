#!/usr/bin/env python

# This script is used in setup.cfg and circleci.
import logging
from subprocess import check_output

LOGGER = logging.getLogger(__name__)


def _run(args):
    return check_output(args).decode('utf8').strip()


def is_git_repo():
    cmd = ["git", "rev-parse", "--is-inside-work-tree"]
    try:
        output = _run(cmd)
        LOGGER.info(f"is git repo cmd output: {output}")
        if output == "true":
            return True
    except Exception as e:
        LOGGER.exception(e)
    return False


def get_version():
    LOGGER.info("Geting version using git.")
    if not is_git_repo():
        # In the production docker image git folder doesn't exist.
        LOGGER.warn("We are not in a git folder, probably running in production, cannot return version.")
        version_returned = 'unknown'
        print(f"version=={version_returned}")
        return version_returned
    describe_cmd = ['git', 'describe', '--tags', '--always']
    last_tag = _run(describe_cmd + ['--abbrev=0'])  # '1.0.14'
    describe = _run(describe_cmd)  # '1.0.14-2-gfaa2442'  {tag}-{nb_commit_since_tag}-{hash}'
    if describe == last_tag:
        version_returned = last_tag.replace('v', '', 1)
        LOGGER.info("version is the last tag, version==%s", version_returned)
        print(f"version=={version_returned}")
        return version_returned
    short_hash = describe[len(last_tag) + 1:].split('-')[1]
    version_returned = "{}.dev0+{}".format(last_tag.replace('v', '', 1), short_hash[1:])
    LOGGER.info("version==%s", version_returned)
    print(f"version=={version_returned}")
    return version_returned


VERSION = get_version() or "local"

if __name__ == '__main__':
    print(VERSION)
