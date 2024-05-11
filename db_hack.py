import streamlit as st
import pathlib
import os


def T(name):
    name = name[name.find("_") + 1 : -3].replace("_", " ")
    return name


def code_explorer():
    code = {}
    this_file = pathlib.Path(__file__)
    if this_file.name == "Home.py":
        app_dir = this_file.parent
    else:
        app_dir = this_file.parent.parent.parent

    home = app_dir / "Home.py"

    code["Page: " + "Home"] = home

    pages = (app_dir / "pages").glob("*.py")
    for p in pages:
        if "🔒" not in p.name:
            name = T(p.name)
            if "Code" not in p.name:
                code["Page: " + name] = p

    libs = (app_dir / "libs").glob("*")
    for l in libs:
        code["Library: " + l.name] = l / "__init__.py"

    jobs = pathlib.Path("/app/src/jobs/").glob("*")
    for j in jobs:
        code["Job: " + j.name] = j / "main.py"

    select = st.selectbox("Pick component", options=code.keys())
    if select:
        st.code(code[select].read_text())