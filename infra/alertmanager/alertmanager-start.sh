#!/usr/bin/env sh
set -eu

CONFIG_PATH="/etc/alertmanager/alertmanager.yml"

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

require_env() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Missing required environment variable for email alerts: $name" >&2
    exit 1
  fi
}

if [ "${ALERT_EMAIL_ENABLED:-false}" = "true" ]; then
  require_env ALERT_SMTP_SMARTHOST
  require_env ALERT_SMTP_FROM
  require_env ALERT_SMTP_AUTH_USERNAME
  require_env ALERT_SMTP_AUTH_PASSWORD
  require_env ALERT_EMAIL_TO

  sed \
    -e "s|smtp.example.com:587|$(escape_sed "$ALERT_SMTP_SMARTHOST")|g" \
    -e "s|alerts-from@example.com|$(escape_sed "$ALERT_SMTP_FROM")|g" \
    -e "s|alerts-user@example.com|$(escape_sed "$ALERT_SMTP_AUTH_USERNAME")|g" \
    -e "s|change-me|$(escape_sed "$ALERT_SMTP_AUTH_PASSWORD")|g" \
    -e "s|student@example.com|$(escape_sed "$ALERT_EMAIL_TO")|g" \
    /etc/alertmanager/alertmanager-email.template.yml > /tmp/alertmanager.yml
  CONFIG_PATH="/tmp/alertmanager.yml"
fi

exec /bin/alertmanager \
  --config.file="$CONFIG_PATH" \
  --storage.path=/alertmanager
