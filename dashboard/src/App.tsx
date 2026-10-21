import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  AlertCircle,
  Check,
  Copy,
  Download,
  FileAudio,
  KeyRound,
  Loader2,
  Moon,
  RefreshCw,
  Sun,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"

type JobSummary = {
  id: string
  status: "queued" | "running" | "done" | "error"
  error: string | null
  created: number
  started: number | null
  finished: number | null
  options: Record<string, string | number>
  progress: string | null
}

type Turn = { speaker: number; start: number; end: number; text: string; name?: string }
type Transcript = { source: string; model: string; language: string; turns: Turn[] }

type Job = JobSummary & {
  json: Transcript | null
  markdown: string | null
  audio: boolean
}

const LANGUAGES = ["auto", "it", "de", "en", "fr", "es"]

// XHR instead of fetch: only XHR reports upload progress.
function upload(url: string, file: File, token: string, onProgress: (fraction: number) => void) {
  return new Promise<JobSummary>((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open("POST", url)
    request.setRequestHeader("Authorization", `Bearer ${token}`)
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total)
    }
    request.onload = () => {
      const body = (() => {
        try {
          return JSON.parse(request.responseText)
        } catch {
          return {}
        }
      })()
      if (request.status >= 200 && request.status < 300) resolve(body)
      else reject(new Error(body.error ?? (request.statusText || `HTTP ${request.status}`)))
    }
    request.onerror = () => reject(new Error("network error"))
    request.send(file)
  })
}

// navigator.clipboard is undefined outside a secure context (plain http on a LAN name).
async function copyText(text: string) {
  try {
    if (navigator.clipboard) return await navigator.clipboard.writeText(text)
  } catch {
    // fall through to the legacy path
  }
  const area = document.createElement("textarea")
  area.value = text
  area.style.position = "fixed"
  area.style.opacity = "0"
  document.body.append(area)
  area.select()
  document.execCommand("copy")
  area.remove()
}

const options = (job: JobSummary) =>
  Object.entries(job.options ?? {})
    .map(([key, value]) => `${key}=${value}`)
    .join(" · ")

function clock(seconds: number) {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return m ? `${m}m ${s}s` : `${s}s`
}

function timecode(seconds: number) {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const mm = String(Math.floor((total % 3600) / 60)).padStart(2, "0")
  const ss = String(total % 60).padStart(2, "0")
  return hours ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`
}

function duration(job: JobSummary, now: number) {
  if (!job.started) return "—"
  return clock((job.finished ?? now / 1000) - job.started)
}

function age(created: number, now: number) {
  const elapsed = now / 1000 - created
  if (elapsed < 60) return "just now"
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)} min ago`
  if (elapsed < 86400) return `${Math.floor(elapsed / 3600)} h ago`
  return new Date(created * 1000).toLocaleDateString()
}

function StatusBadge({ status }: { status: JobSummary["status"] }) {
  if (status === "done")
    return (
      <Badge variant="secondary">
        <Check className="size-3" />
        done
      </Badge>
    )
  if (status === "error")
    return (
      <Badge variant="destructive">
        <AlertCircle className="size-3" />
        error
      </Badge>
    )
  if (status === "running")
    return (
      <Badge variant="outline">
        <span className="size-1.5 animate-pulse rounded-full bg-current" />
        running
      </Badge>
    )
  return (
    <Badge variant="outline" className="text-muted-foreground">
      queued
    </Badge>
  )
}

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("meeting-token") ?? "")
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [now, setNow] = useState(Date.now())
  const [loading, setLoading] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [sent, setSent] = useState<{ name: string; fraction: number } | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Job | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const api = useCallback(
    async (path: string, init: RequestInit = {}) => {
      const response = await fetch(path, {
        ...init,
        headers: { Authorization: `Bearer ${token}`, ...init.headers },
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error ?? response.statusText)
      return body
    },
    [token]
  )

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { jobs } = await api("/jobs")
      setJobs(jobs)
      setAuthError(null)
      // Keep an open transcript live: progress lines and the final markdown.
      if (selectedId) setSelected(await api(`/jobs/${selectedId}`))
    } catch (error) {
      setAuthError((error as Error).message)
    } finally {
      setLoading(false)
      setNow(Date.now())
    }
  }, [api, token, selectedId])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 5000)
    return () => clearInterval(timer)
  }, [refresh])

  async function openJob(id: string) {
    setSelectedId(id)
    try {
      setSelected(await api(`/jobs/${id}`))
    } catch (error) {
      setAuthError((error as Error).message)
    }
  }

  function closeJob() {
    setSelectedId(null)
    setSelected(null)
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const file = form.get("file") as File | null
    if (!file?.size) return
    const query = new URLSearchParams({ filename: file.name })
    const speakers = String(form.get("speakers") ?? "")
    const language = String(form.get("language") ?? "auto")
    if (speakers) query.set("speakers", speakers)
    if (language !== "auto") query.set("language", language)

    setUploading(true)
    setUploadError(null)
    setSent({ name: file.name, fraction: 0 })
    try {
      const job = await upload(`/jobs?${query}`, file, token, (fraction) =>
        setSent({ name: file.name, fraction })
      )
      if (fileInput.current) fileInput.current.value = ""
      await refresh()
      await openJob(job.id)
    } catch (error) {
      setUploadError((error as Error).message)
    } finally {
      setUploading(false)
      setSent(null)
    }
  }

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-10">
      <header className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">meeting-stack</h1>
          <p className="text-sm text-muted-foreground">
            Diarized transcripts. Upload a recording, read the result.
          </p>
        </div>
        <ThemeToggle />
      </header>

      <Card>
        <CardHeader>
          <CardTitle>New transcription</CardTitle>
          <CardDescription>
            The file is queued and transcribed in the background.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <form onSubmit={submit} className="space-y-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="grid flex-1 gap-2">
                <Label htmlFor="file">Recording</Label>
                <Input
                  id="file"
                  name="file"
                  type="file"
                  accept="audio/*,video/*"
                  required
                  ref={fileInput}
                />
              </div>
              <div className="grid w-full gap-2 sm:w-24">
                <Label htmlFor="speakers">Speakers</Label>
                <Input
                  id="speakers"
                  name="speakers"
                  type="number"
                  min={1}
                  max={20}
                  placeholder="auto"
                />
              </div>
              <div className="grid w-full gap-2 sm:w-36">
                <Label htmlFor="language">Language</Label>
                <Select name="language" defaultValue="auto">
                  <SelectTrigger id="language" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGES.map((code) => (
                      <SelectItem key={code} value={code}>
                        {code === "auto" ? "Auto-detect" : code.toUpperCase()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button type="submit" disabled={uploading}>
                {uploading ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <FileAudio />
                )}
                {uploading ? "Uploading…" : "Transcribe"}
              </Button>
            </div>

            {sent && (
              <div className="space-y-2" aria-live="polite">
                <div className="flex items-baseline justify-between gap-4 text-xs">
                  <span className="truncate text-muted-foreground">{sent.name}</span>
                  <span className="font-mono tabular-nums text-muted-foreground">
                    {sent.fraction < 1
                      ? `Uploading ${Math.round(sent.fraction * 100)}%`
                      : "Queued, waiting for the server…"}
                  </span>
                </div>
                <Progress value={sent.fraction * 100} />
              </div>
            )}

            {uploadError && (
              <p className="flex items-center gap-2 text-sm text-destructive" role="alert">
                <AlertCircle className="size-4" />
                {uploadError}
              </p>
            )}
          </form>

          <Separator />

          <div className="flex flex-col gap-2">
            <Label htmlFor="token" className="flex items-center gap-2">
              <KeyRound className="size-3.5 text-muted-foreground" />
              Bearer token
            </Label>
            <Input
              id="token"
              type="password"
              autoComplete="off"
              placeholder="MEETING_TOKEN"
              value={token}
              onChange={(event) => {
                setToken(event.target.value)
                localStorage.setItem("meeting-token", event.target.value)
              }}
              className="sm:max-w-sm"
            />
            <p className={cn("text-xs", authError ? "text-destructive" : "text-muted-foreground")}>
              {authError ??
                "Sent as Authorization: Bearer, kept in this browser only. Leave empty if the service runs without a token."}
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="py-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 border-b py-4">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2">
              Jobs
              <Badge variant="outline" className="font-mono">
                {jobs.length}
              </Badge>
            </CardTitle>
            <CardDescription>Updated every 5 s.</CardDescription>
          </div>
          <Button variant="ghost" size="icon" onClick={refresh} disabled={loading} title="Refresh">
            <RefreshCw className={cn(loading && "animate-spin")} />
          </Button>
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Created</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Length</TableHead>
                <TableHead>Job</TableHead>
                <TableHead className="pr-6">Progress</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="h-32 text-center text-muted-foreground">
                    No jobs yet.
                  </TableCell>
                </TableRow>
              ) : (
                jobs
                  .slice()
                  .sort((a, b) => b.created - a.created)
                  .map((job) => (
                    <TableRow
                      key={job.id}
                      onClick={() => openJob(job.id)}
                      className="cursor-pointer"
                    >
                      <TableCell className="pl-6 text-muted-foreground">
                        {age(job.created, now)}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={job.status} />
                      </TableCell>
                      <TableCell className="font-mono text-xs tabular-nums">
                        {duration(job, now)}
                      </TableCell>
                      <TableCell>
                        <div className="font-mono text-xs">{job.id.slice(0, 8)}</div>
                        <div className="max-w-52 truncate text-xs text-muted-foreground">
                          {options(job) || "defaults"}
                        </div>
                      </TableCell>
                      <TableCell className="pr-6">
                        <div
                          className={cn(
                            "max-w-56 truncate text-xs",
                            job.status === "error" ? "text-destructive" : "text-muted-foreground"
                          )}
                          title={job.error ?? job.progress ?? undefined}
                        >
                          {job.error ??
                            (job.status === "done" ? "—" : job.progress ?? "—")}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <TranscriptDialog job={selected} token={token} onClose={closeJob} />
    </div>
  )
}

function TranscriptDialog({
  job,
  token,
  onClose,
}: {
  job: Job | null
  token: string
  onClose: () => void
}) {
  const [copied, setCopied] = useState(false)
  const [active, setActive] = useState(-1)
  const [seekTick, setSeekTick] = useState(0)
  const [audioFailed, setAudioFailed] = useState(false)
  const audio = useRef<HTMLAudioElement | null>(null)
  const activeLine = useRef<HTMLButtonElement | null>(null)
  const scroller = useRef<HTMLDivElement | null>(null)
  const seeked = useRef(false)

  const markdown = job?.markdown ?? ""
  const turns = useMemo(() => job?.json?.turns ?? [], [job?.json])
  const downloadUrl = useMemo(
    () => (markdown ? URL.createObjectURL(new Blob([markdown], { type: "text/markdown" })) : ""),
    [markdown]
  )

  useEffect(() => () => void (downloadUrl && URL.revokeObjectURL(downloadUrl)), [downloadUrl])
  useEffect(() => {
    setCopied(false)
    setActive(-1)
    setAudioFailed(false)
  }, [job?.id])

  const follow = useCallback(() => {
    const element = audio.current
    if (!element) return
    const time = element.currentTime
    // The turn being spoken, and through a silence the one just spoken: keeping
    // it lit reads better than blinking on and off between turns.
    let index = -1
    for (let i = 0; i < turns.length && time >= turns[i].start; i++) index = i
    setActive(index)
  }, [turns])

  // After the render that moves the highlight, so the ref points at the new line.
  // Playback drags the view along only once the line leaves the upper half --
  // otherwise it would scroll on every turn -- but a seek always scrolls, even
  // paused, because the reader asked for that spot. A paused reader who is not
  // seeking keeps their place.
  useEffect(() => {
    const line = activeLine.current
    const box = scroller.current
    if (active === -1 || !line || !box) return
    if (audio.current?.paused && !seeked.current) return
    seeked.current = false
    const offset = line.getBoundingClientRect().top - box.getBoundingClientRect().top
    if (offset < 0 || offset > box.clientHeight * 0.55)
      box.scrollTo({
        top: box.scrollTop + offset - box.clientHeight * 0.25,
        behavior: "smooth",
      })
  }, [active, seekTick])

  useEffect(() => {
    const element = audio.current
    if (!element) return
    const onSeek = () => {
      // A seek is a jump, even while paused: the view must follow it.
      seeked.current = true
      follow()
      setSeekTick((tick) => tick + 1)
    }
    element.addEventListener("timeupdate", follow)
    element.addEventListener("seeked", onSeek)
    return () => {
      element.removeEventListener("timeupdate", follow)
      element.removeEventListener("seeked", onSeek)
    }
  }, [follow, job?.id, audioFailed])

  async function copy() {
    await copyText(markdown)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  function playFrom(turn: Turn) {
    const element = audio.current
    if (!element) return
    element.currentTime = turn.start
    void element.play().catch(() => {}) // autoplay may be refused; the seek stands
  }

  const source = job
    ? `/jobs/${job.id}/audio${token ? `?token=${encodeURIComponent(token)}` : ""}`
    : ""

  return (
    <Dialog open={job !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="flex max-h-[88vh] flex-col gap-0 p-0 sm:max-w-3xl">
        <DialogHeader className="space-y-1 border-b p-6 pb-4">
          <DialogTitle className="flex items-center gap-2">
            {job?.status === "done" ? `Transcript · ${job.id.slice(0, 8)}` : `Job · ${job?.id.slice(0, 8)}`}
            {job && <StatusBadge status={job.status} />}
          </DialogTitle>
          <DialogDescription className="font-mono text-xs">
            {job ? options(job) || "default options" : ""}
            {turns.length ? ` · ${turns.length} turns` : ""}
          </DialogDescription>
        </DialogHeader>

        {job?.status === "done" && (
          <div className="border-b px-6 py-3">
            {job.audio && !audioFailed ? (
              <audio
                key={job.id}
                ref={audio}
                controls
                preload="metadata"
                className="w-full accent-primary"
                src={source}
                onError={() => setAudioFailed(true)}
              />
            ) : (
              <p className="text-xs text-muted-foreground">
                {audioFailed
                  ? "The recording could not be loaded."
                  : "No recording kept for this job — it ran before --keep-audio was on."}
              </p>
            )}
          </div>
        )}

        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
          {job?.status === "done" ? (
            turns.length ? (
              <ol className="space-y-0.5 p-4">
                {turns.map((turn, index) => (
                  <li key={index}>
                    <button
                      type="button"
                      ref={index === active ? activeLine : undefined}
                      onClick={() => playFrom(turn)}
                      className={cn(
                        "block w-full rounded-md px-2 py-1 text-left transition-colors hover:bg-accent/60",
                        index === active && "bg-primary text-primary-foreground"
                      )}
                    >
                      <span
                        className={cn(
                          "mr-2 font-mono text-xs tabular-nums",
                          index === active ? "text-primary-foreground/70" : "text-muted-foreground"
                        )}
                      >
                        {timecode(turn.start)}
                      </span>
                      <span className="text-sm font-medium">
                        {turn.name ?? `Speaker ${turn.speaker}`}
                      </span>{" "}
                      <span className="text-sm">{turn.text}</span>
                    </button>
                  </li>
                ))}
              </ol>
            ) : (
              <pre className="p-6 font-mono text-[13px] leading-relaxed whitespace-pre-wrap">
                {markdown}
              </pre>
            )
          ) : (
            <div className="space-y-2 p-6">
              <p className="text-sm text-muted-foreground">
                {job?.status === "error" ? "The job failed:" : "Still working on it:"}
              </p>
              <pre
                className={cn(
                  "rounded-lg bg-muted p-4 font-mono text-xs whitespace-pre-wrap",
                  job?.status === "error" && "text-destructive"
                )}
              >
                {job?.error ?? job?.progress ?? "queued"}
              </pre>
            </div>
          )}
        </div>

        {job?.status === "done" && (
          <DialogFooter className="flex-row justify-end border-t p-4">
            <>
              <Button variant="outline" size="sm" onClick={copy}>
                {copied ? <Check /> : <Copy />}
                {copied ? "Copied" : "Copy"}
              </Button>
              <Button variant="outline" size="sm" asChild>
                <a href={downloadUrl} download={`${job.id.slice(0, 8)}.md`}>
                  <Download />
                  Download .md
                </a>
              </Button>
            </>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}

function ThemeToggle() {
  const [dark, setDark] = useState(
    () => document.documentElement.classList.contains("dark")
  )

  function toggle() {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle("dark", next)
    localStorage.setItem("theme", next ? "dark" : "light")
  }

  return (
    <Button variant="ghost" size="icon" onClick={toggle} title="Toggle theme">
      {dark ? <Sun /> : <Moon />}
    </Button>
  )
}
