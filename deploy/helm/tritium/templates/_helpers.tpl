{{- define "tritium.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "tritium.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "tritium.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "tritium.labels" -}}
app.kubernetes.io/name: {{ include "tritium.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
tritium.ai/backend: {{ .Values.backend | quote }}
{{- end -}}

{{- define "tritium.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tritium.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
