import { useCallback, useEffect, useRef, useState } from "react"
import { AlertCircle, ChevronDown, ChevronUp, KeyRound, Mic, Monitor, Square } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type Row = { i: number; src: string; mt: string; done?: boolean; error?: string }

const LANGUAGES = ["auto", "de", "it", "en", "fr", "es"]
const TARGETS = ["it", "en", "de", "fr"]
const CHUNK = 8000 // samples: half a second of 16 kHz audio per request

/**
 * The microphone's samples as 16 kHz, whatever rate the browser gave us.
 *
 * Asking for `new AudioContext({ sampleRate: 16000 })` is not portable: Gecko
 * often hands back the device rate (48 kHz) instead, and refusing to run in that
 * case left Firefox and Zen with an open microphone and no text at all. Down
 * sampling averages the samples of each output step, which is a crude low-pass —
 * enough for speech, and it keeps the pitch right.
 */
function toSixteen(input: Float32Array, rate: number): Int16Array {
  const step = rate / 16000
  const length = Math.max(1, Math.floor(input.length / step))
  const pcm = new Int16Array(length)
  for (let index = 0; index < length; index++) {
    let value: number
    if (step <= 1) {
      value = input[Math.min(index, input.length - 1)]
    } else {
      const from = Math.floor(index * step)
      const to = Math.min(Math.floor((index + 1) * step), input.length)
      let sum = 0
      for (let at = from; at < to; at++) sum += input[at]
      value = sum / Math.max(1, to - from)
    }
    value = Math.max(-1, Math.min(1, value))
    pcm[index] = value < 0 ? value * 0x8000 : value * 0x7fff
  }
  return pcm
}

/** A setting the page remembers: the next meeting starts where this one ended. */
function useStored<T>(key: string, initial: T) {
  const name = `live-${key}`
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(name)
      return stored === null ? initial : (JSON.parse(stored) as T)
    } catch {
      return initial
    }
  })
  useEffect(() => {
    localStorage.setItem(name, JSON.stringify(value))
  }, [name, value])
  return [value, setValue] as const
}

/**
 * Live transcription and translation of a meeting as it happens.
 *
 * The browser samples at 16 kHz itself (`AudioContext({ sampleRate: 16000 })`),
 * so nothing is resampled here; half a second goes out at a time, the server
 * cuts lines on pauses and on changes of voice, and this view only draws what
 * the session reports — the meter and the state badge are there because ten
 * silent seconds of speech otherwise look like a broken page.
 */
export default function Live() {
  const [token, setToken] = useState(
    () => new URLSearchParams(location.search).get("token") ?? localStorage.getItem("meeting-token") ?? ""
  )
  const [language, setLanguage] = useStored("language", "de")
  const [target, setTarget] = useStored("target", "it")
  const [device, setDevice] = useStored("device", "")
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [processing, setProcessing] = useStored("processing", true)
  const [rows, setRows] = useState<Row[]>([])
  const [state, setState] = useState("idle")
  const [buffered, setBuffered] = useState(0)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [started, setStarted] = useState(false)
  const [expanded, setExpanded] = useStored("expanded", false)
  const [error, setError] = useState<string | null>(null)

  const session = useRef<string | null>(null)
  const cursor = useRef(0)
  const chain = useRef<Promise<unknown>>(Promise.resolve())
  const context = useRef<AudioContext | null>(null)
  const mix = useRef<GainNode | null>(null)
  const streams = useRef<MediaStream[]>([])
  const samples = useRef<Int16Array[]>([])
  const queued = useRef(0)
  const level = useRef(0)
  const list = useRef<HTMLDivElement>(null)
  const tail = useRef(true) // follow the newest line unless the reader scrolled up

  const deviceLabel = devices.find((entry) => entry.deviceId === device)?.label ?? "Default input"

  const auth = useCallback(
    () => ({ Authorization: `Bearer ${token}` }),
    [token]
  )

  // Inputs are listed once the page has the microphone permission: before that
  // the labels are empty, and an empty list is not a choice anybody can make.
  const listDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    try {
      const all = await navigator.mediaDevices.enumerateDevices()
      const inputs = all.filter((entry) => entry.kind === "audioinput" && entry.label)
      setDevices(inputs)
      // A remembered device that is no longer plugged in is not a choice.
      setDevice((current) =>
        current && !inputs.some((entry) => entry.deviceId === current) ? "" : current
      )
    } catch {
      // no permission yet: the default input is the only offer
    }
  }, [])

  useEffect(() => {
    listDevices()
    navigator.mediaDevices?.addEventListener?.("devicechange", listDevices)
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", listDevices)
  }, [listDevices])

  // Reading the session: the cursor the server returns stops at the first row
  // still being translated, so following it is enough to see a row grow.
  useEffect(() => {
    if (!sessionId) return
    let alive = true
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`/live/${sessionId}?since=${cursor.current}`, {
          headers: auth(),
        })
        if (response.status === 401) {
          setError("missing or wrong bearer token")
          return
        }
        const view = await response.json()
        if (!alive) return
        // 404 is the half second between `begin()` and the first chunk, which is
        // what creates the session on the server. Not an error, not a row.
        if (response.status === 404) return
        if (!response.ok) {
          setError(view.error ?? response.statusText)
          return
        }
        cursor.current = view.cursor
        setState(view.state)
        setBuffered(view.buffered)
        const fresh: Row[] = Array.isArray(view.rows) ? view.rows : []
        // Never throw from inside an updater: React answers by unmounting the
        // tree, and a blank page is the least informative failure there is.
        if (!fresh.length) return
        setRows((previous) => {
          const next = previous.slice()
          for (const row of fresh) next[row.i] = row
          return next
        })
      } catch (failure) {
        setError((failure as Error).message)
      }
    }, 400)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [sessionId, auth])

  // Follow the tail, but never steal the scroll from someone reading back.
  useEffect(() => {
    const element = list.current
    if (element && tail.current) element.scrollTop = element.scrollHeight
  }, [rows])

  function watching() {
    const element = list.current
    if (!element) return
    tail.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48
  }

  function graph(): boolean {
    if (context.current) return true
    const created = new AudioContext()
    const gain = created.createGain()
    const node = created.createScriptProcessor(1024, 1, 1) // 64 ms: a meter that moves
    node.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      event.outputBuffer.getChannelData(0).fill(0) // this node is only a tap
      let peak = 0
      for (let index = 0; index < input.length; index++) {
        const value = Math.abs(input[index])
        if (value > peak) peak = value
      }
      level.current = Math.min(1, peak * 0x7fff / 8000) // speech peaks around 1000-8000
      const pcm = toSixteen(input, created.sampleRate)
      samples.current.push(pcm)
      queued.current += pcm.length
      if (queued.current >= CHUNK) flush()
    }
    gain.connect(node)
    node.connect(created.destination) // Chrome runs a node only when it is connected
    context.current = created
    mix.current = gain
    return true
  }

  function flush() {
    const current = session.current
    if (!current || !queued.current) return
    const bytes = new Uint8Array(queued.current * 2)
    let at = 0
    for (const pcm of samples.current) {
      bytes.set(new Uint8Array(pcm.buffer), at)
      at += pcm.length * 2
    }
    samples.current = []
    queued.current = 0
    const query = new URLSearchParams({ language, to: target })
    chain.current = chain.current
      .then(() =>
        fetch(`/live/${current}/audio?${query}`, {
          method: "POST",
          body: bytes,
          headers: auth(),
        })
      )
      .then((response) => response.json())
      .then((view) => {
        if (!view.ok) {
          setError(view.error ?? "the server refused the audio")
          if (!view.open) stop()
        }
      })
      .catch((failure) => setError((failure as Error).message))
  }

  async function addStream(stream: MediaStream) {
    if (!mix.current || !context.current) return
    streams.current.push(stream)
    context.current.createMediaStreamSource(stream).connect(mix.current)
    if (context.current.state === "suspended") await context.current.resume()
  }

  function begin() {
    const id = crypto.randomUUID?.() ?? `live-${Date.now()}`
    session.current = id
    cursor.current = 0
    setSessionId(id)
    setStarted(true)
    setError(null)
  }

  // Plain http:// to another host is not a secure context, and there
  // navigator.mediaDevices does not exist at all — the failure would otherwise
  // read "can't access property getUserMedia", which says nothing about why.
  // Both capture paths need it: getUserMedia and getDisplayMedia alike.
  function noCapture(): boolean {
    if (navigator.mediaDevices) return false
    setError(
      `${window.location.origin} is not a secure context, so this browser exposes no ` +
        "microphone and no screen capture. Open the page as http://127.0.0.1:8099/live " +
        "on the box, through an SSH tunnel, or behind https: plain http:// to another " +
        "host never works, in any browser."
    )
    return true
  }

  async function startMicrophone() {
    if (noCapture()) return
    if (!graph()) return
    try {
      begin()
      // A virtual loopback must not be filtered: noise suppression eats exactly
      // the quiet half of the other person's voice.
      await addStream(
        await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            ...(processing
              ? { echoCancellation: true, noiseSuppression: true }
              : { echoCancellation: false, noiseSuppression: false, autoGainControl: false }),
            ...(device ? { deviceId: { exact: device } } : {}),
          },
        })
      )
      listDevices()
    } catch (failure) {
      setError(`microphone: ${(failure as Error).message}`)
      stop()
    }
  }

  async function startSystem() {
    if (noCapture()) return
    if (/Firefox\//.test(navigator.userAgent)) {
      setError(
        "This browser (Firefox, and Zen on top of it) has no audio in screen sharing: " +
          "getDisplayMedia there is video only. Pick a virtual input device above instead " +
          "(BlackHole on macOS, a PipeWire monitor on Linux), with voice filters off."
      )
      return
    }
    if (!graph()) return
    try {
      begin()
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true })
      stream.getVideoTracks().forEach((track) => track.stop())
      if (!stream.getAudioTracks().length) {
        setError(
          "No audio in that share: Chrome never takes audio from a window. Share a tab, " +
            "or the whole screen with system audio (Chrome 141+ on macOS)."
        )
        stop()
        return
      }
      await addStream(stream)
    } catch (failure) {
      setError(`system audio: ${(failure as Error).message}`)
      stop()
    }
  }

  function stop() {
    const current = session.current
    flush() // the last partial second is still words
    streams.current.forEach((stream) => stream.getTracks().forEach((track) => track.stop()))
    streams.current = []
    context.current?.close().catch(() => {})
    context.current = null
    mix.current = null
    level.current = 0
    session.current = null
    setStarted(false)
    if (!current) {
      setSessionId(null)
      return
    }
    // Close, then keep reading: the last line is still being transcribed, and
    // the session is forgotten by the server a while after it drains.
    chain.current = chain.current
      .then(() => fetch(`/live/${current}/close`, { method: "POST", headers: auth() }))
      .catch(() => {})
    setTimeout(() => setSessionId((id) => (id === current ? null : id)), 20000)
  }

  return (
    // A column that fills the window: the controls take what they need and the
    // transcript gets the rest, so "the last few lines" means the same thing on
    // a laptop and on a monitor instead of depending on a hard-coded height.
    <div className="mx-auto flex h-[calc(100svh-3.25rem)] w-full max-w-5xl flex-col gap-6 overflow-y-auto px-4 py-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            Live
            <SessionBadge state={state} started={started} buffered={buffered} />
          </CardTitle>
          <CardDescription>
            A line ends on a pause or when the voice changes. No speaker labels here: the
            batch transcript is the labelled truth.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Two rows on purpose: the buttons are one height, the labelled
              columns are another, and mixing them made the meter sit on the floor
              of the row instead of beside the buttons. */}
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={startMicrophone} disabled={started}>
              <Mic />
              Microphone
            </Button>
            <Button
              variant="secondary"
              onClick={startSystem}
              disabled={started}
              title="Chrome shares a tab's audio, never a window's. Firefox and Zen have no display audio at all: pick a virtual input device (BlackHole, a PipeWire monitor) and turn voice filters off."
            >
              <Monitor />
              System audio
            </Button>
            <Button variant="outline" onClick={stop} disabled={!started}>
              <Square />
              Stop
            </Button>
            <Meter level={level} live={started} />
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="device" className="text-xs">
                Input
              </Label>
              {/* The children are not decoration: Radix copies an item's text into
                  the trigger through a portal only when the trigger has none, and
                  with none the copy was made for every item that had mounted, so the
                  trigger read "Default inputFake Default Audio Input". */}
              <Select
                value={device || "default"}
                onValueChange={(value) => setDevice(value === "default" ? "" : value)}
                disabled={started}
              >
                <SelectTrigger id="device" className="w-[200px]">
                  <SelectValue>{deviceLabel}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default input</SelectItem>
                  {devices.map((entry) => (
                    <SelectItem key={entry.deviceId} value={entry.deviceId}>
                      {entry.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-1.5">
              <Label className="text-xs">Voice filters</Label>
              <Button
                variant={processing ? "secondary" : "outline"}
                onClick={() => setProcessing((value) => !value)}
                disabled={started}
                title="Echo cancellation and noise suppression: turn them off for a virtual device"
              >
                {processing ? "On" : "Off"}
              </Button>
            </div>

            <div className="grid gap-1.5">
              <Label htmlFor="language" className="text-xs">
                Spoken
              </Label>
              <Select value={language} onValueChange={setLanguage} disabled={started}>
                <SelectTrigger id="language" className="w-24">
                  <SelectValue>{language}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {LANGUAGES.map((code) => (
                    <SelectItem key={code} value={code}>
                      {code}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-1.5">
              <Label htmlFor="target" className="text-xs">
                Translate into
              </Label>
              <Select value={target} onValueChange={setTarget} disabled={started}>
                <SelectTrigger id="target" className="w-24">
                  <SelectValue>{target}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {TARGETS.map((code) => (
                    <SelectItem key={code} value={code}>
                      {code}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid min-w-[220px] flex-1 gap-1.5">
              <Label htmlFor="token" className="flex items-center gap-2 text-xs">
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
              />
            </div>
          </div>

          {error && (
            <p className="flex items-start gap-2 text-sm text-destructive" role="alert">
              <AlertCircle className="mt-0.5 size-4 shrink-0" />
              {error}
            </p>
          )}
        </CardContent>
      </Card>

      {rows.length > 0 ? (
        <Card
          className={cn(
            "flex min-h-0 flex-col overflow-hidden",
            expanded ? "flex-1" : "flex-none"
          )}
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <CardTitle className="text-base">Transcript</CardTitle>
            <Button variant="ghost" size="sm" onClick={() => setExpanded((value) => !value)}>
              {expanded ? <ChevronUp /> : <ChevronDown />}
              {expanded ? "Collapse" : "Expand"}
            </Button>
          </CardHeader>
          {/* Three lines are the tail you read while listening; Expand opens the
              whole thing. The newest line is full, the ones above step back. */}
          <div
            ref={list}
            onScroll={watching}
            className={cn(
              "flex snap-y snap-proximity flex-col overflow-y-auto",
              expanded && "min-h-0 flex-1"
            )}
          >
            {/* `mt-auto`: with few lines they sit at the bottom of the window,
                which is where the newest one belongs — the empty space goes up
                under the gradient instead of hanging below the text. */}
            <div className="mt-auto flex flex-col">
              {(expanded ? rows : rows.slice(-3)).map((row) => (
                <div
                  key={row.i}
                  className="grid snap-start gap-1 border-b px-6 py-4 last:border-b-0 sm:grid-cols-2 sm:gap-6"
                  style={{ opacity: fade(rows.length - 1 - row.i) }}
                >
                  <p className="whitespace-pre-wrap text-sm text-muted-foreground">
                    {row.src}
                  </p>
                  <p className="whitespace-pre-wrap text-sm">
                    {row.error ? (
                      <span className="text-destructive">{row.error}</span>
                    ) : (
                      row.mt || "…"
                    )}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </Card>
      ) : (
        <p className="text-sm text-muted-foreground">
          Nothing yet. A line appears as soon as the speaker pauses — or someone else starts
          talking, which the voice embedding also catches.
        </p>
      )}
    </div>
  )
}

// How the tail reads: the newest line full, the ones above it stepping back. A
// mask on the container faded nothing until the rows overflowed it, and this is
// also true with four lines on screen.
const FADE = [1, 0.86, 0.72, 0.58, 0.44, 0.3]

function fade(distance: number) {
  return FADE[Math.min(Math.max(distance, 0), FADE.length - 1)]
}

function SessionBadge({
  state,
  started,
  buffered,
}: {
  state: string
  started: boolean
  buffered: number
}) {
  if (!started)
    return (
      <Badge variant="outline" className="text-muted-foreground">
        <span className="size-1.5 rounded-full bg-current" />
        idle
      </Badge>
    )
  if (state === "asr" || state === "mt")
    return (
      <Badge variant="outline">
        <span className="size-1.5 animate-pulse rounded-full bg-current" />
        {state === "asr" ? "transcribing" : "translating"}
      </Badge>
    )
  return (
    <Badge variant="secondary" className="tabular-nums">
      listening · {buffered.toFixed(1)} s
    </Badge>
  )
}

/**
 * The level, kept out of the rows: it changes fifteen times a second, and a
 * re-render of the transcript at that rate would be a waste of everything.
 */
function Meter({ level, live }: { level: React.RefObject<number>; live: boolean }) {
  const [shown, setShown] = useState(0)

  useEffect(() => {
    if (!live) {
      setShown(0)
      return
    }
    let falling = 0
    const timer = setInterval(() => {
      falling = Math.max(level.current, falling * 0.82) // a fall, not a snap
      setShown(falling)
    }, 80)
    return () => clearInterval(timer)
  }, [live, level])

  return (
    <div className="flex h-9 items-center gap-2">
      <span className="text-xs text-muted-foreground">mic</span>
      <Progress value={shown * 100} className="w-24" />
    </div>
  )
}
