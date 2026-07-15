
"""
Module to expose more detailed version info for the installed `scipy`
"""
version = "1.13.0.dev0+git20260710.6da69cc"
full_version = version
short_version = version.split('.dev')[0]
git_revision = "6da69ccbf8fddb7efdecad5d3e3e2bab21258371"
release = 'dev' not in version and '+' not in version

if not release:
    version = full_version
