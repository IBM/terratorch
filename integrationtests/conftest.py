def pytest_addoption(parser):
    parser.addoption(
        "--env",
        action="store",
        default="ccc",
        choices=[ "ccc", "oc"],
        help="Specify test environment: ccc or openshift",
    )
