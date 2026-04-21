#!/usr/bin/env zsh

if [[ -n "${__PFE_AUTO_ACTIVATE_LOADED:-}" ]]; then
  return 0
fi

typeset -g __PFE_AUTO_ACTIVATE_LOADED=1
typeset -g __PFE_AUTO_ACTIVATE_SCRIPT="${${(%):-%N}:A}"
typeset -g __PFE_AUTO_ACTIVATE_ROOT="${__PFE_AUTO_ACTIVATE_SCRIPT:h:h}"
typeset -g __PFE_AUTO_ACTIVATE_VENV="${__PFE_AUTO_ACTIVATE_ROOT}/.venv"

autoload -Uz add-zsh-hook

function _pfe_auto_activate_is_function() {
  local name="${1:-}"
  [[ -n "$name" ]] || return 1
  [[ "$(type -w "$name" 2>/dev/null)" == *": function" ]]
}

function _pfe_auto_activate_chpwd() {
  local repo_root="${__PFE_AUTO_ACTIVATE_ROOT}"
  local repo_prefix="${repo_root}/"
  local venv_root="${__PFE_AUTO_ACTIVATE_VENV}"

  if [[ "$PWD/" == "$repo_prefix"* ]]; then
    if [[ -x "${venv_root}/bin/python" && "${VIRTUAL_ENV:-}" != "$venv_root" ]]; then
      if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV:-}" != "$venv_root" ]] && _pfe_auto_activate_is_function deactivate; then
        deactivate >/dev/null 2>&1 || true
      fi
      source "${venv_root}/bin/activate"
    fi
    return 0
  fi

  if [[ "${VIRTUAL_ENV:-}" == "$venv_root" ]] && _pfe_auto_activate_is_function deactivate; then
    deactivate >/dev/null 2>&1 || true
  fi
}

add-zsh-hook chpwd _pfe_auto_activate_chpwd
_pfe_auto_activate_chpwd
