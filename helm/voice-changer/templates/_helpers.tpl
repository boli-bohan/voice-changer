{{/*
Expand the name of the chart.
*/}}
{{- define "voice-changer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "voice-changer.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "voice-changer.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "voice-changer.labels" -}}
helm.sh/chart: {{ include "voice-changer.chart" . }}
{{ include "voice-changer.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "voice-changer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "voice-changer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API component labels
*/}}
{{- define "voice-changer.api.labels" -}}
{{ include "voice-changer.labels" . }}
app: {{ .Values.global.appName }}
component: {{ .Values.api.name }}
{{- end }}

{{/*
API selector labels
*/}}
{{- define "voice-changer.api.selectorLabels" -}}
app: {{ .Values.global.appName }}
component: {{ .Values.api.name }}
{{- end }}

{{/*
Worker component labels
*/}}
{{- define "voice-changer.worker.labels" -}}
{{ include "voice-changer.labels" . }}
app: {{ .Values.global.appName }}
component: {{ .Values.worker.name }}
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "voice-changer.worker.selectorLabels" -}}
app: {{ .Values.global.appName }}
component: {{ .Values.worker.name }}
{{- end }}

{{/*
Frontend component labels
*/}}
{{- define "voice-changer.frontend.labels" -}}
{{ include "voice-changer.labels" . }}
app: {{ .Values.global.appName }}
component: {{ .Values.frontend.name }}
{{- end }}

{{/*
Frontend selector labels
*/}}
{{- define "voice-changer.frontend.selectorLabels" -}}
app: {{ .Values.global.appName }}
component: {{ .Values.frontend.name }}
{{- end }}

{{/*
API full name
*/}}
{{- define "voice-changer.api.fullname" -}}
{{ .Values.global.appName }}-{{ .Values.api.name }}
{{- end }}

{{/*
Worker full name
*/}}
{{- define "voice-changer.worker.fullname" -}}
{{ .Values.global.appName }}-{{ .Values.worker.name }}
{{- end }}

{{/*
Frontend full name
*/}}
{{- define "voice-changer.frontend.fullname" -}}
{{ .Values.global.appName }}-{{ .Values.frontend.name }}
{{- end }}
