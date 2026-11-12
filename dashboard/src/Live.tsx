import { useCallback, useEffect, useRef, useState } from "react"
import { AlertCircle, KeyRound, Mic, Monitor, Square } from "lucide-react"
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
  const [language, setLanguage] = useState("de")
  const [target, setTarget] = useState("it")
  const [device, setDevice] = useState("")
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [processing, setProcessing] = useState(true)
  const [rows, setRows] = useState<Row[]>([])
  const [state, setState] = useState("idle")
  const [buffered, setBuffered] = useState(0)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [started, setStarted] = useState(false)
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
      setDevices(all.filter((entry) => entry.kind === "audioinput" && entry.label))
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

  function graph(): boolean {
    if (context.current) return true
    const created = new AudioContext({ sampleRate: 16000 })
    if (created.sampleRate !== 16000) {
      created.close().catch(() => {})
      setError(
        "This browser will not give a 16 kHz AudioContext: use Chrome or Firefox, or the line would be pitched."
      )
      return false
    }
    const gain = created.createGain()
    const node = created.createScriptProcessor(1024, 1, 1) // 64 ms: a meter that moves
    node.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      event.outputBuffer.getChannelData(0).fill(0) // this node is only a tap
      const pcm = new Int16Array(input.length)
      let peak = 0
      for (let index = 0; index < input.length; index++) {
        const value = Math.max(-1, Math.min(1, input[index]))
        pcm[index] = value < 0 ? value * 0x8000 : value * 0x7fff
        if (Math.abs(pcm[index]) > peak) peak = Math.abs(pcm[index])
      }
      level.current = Math.min(1, peak / 8000) // speech peaks around 1000-8000
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

  async function startMicrophone() {
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
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            Live
            <SessionBadge state={state} started={started} buffered={buffered} />
          </CardTitle>
          <CardDescription>
            Streaming transcription and translation. A line ends on a pause or when the voice
            changes; there are no speaker labels here — the batch transcript is the labelled
            truth. In Chrome you can add the system audio by sharing a tab, never a window;
            in Firefox and Zen, and for a call in a desktop app, pick a virtual input device
            (BlackHole, a PipeWire monitor) and turn voice filters off.
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
            <Button variant="secondary" onClick={startSystem} disabled={started}>
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
        <Card>
          <CardContent className="divide-y p-0">
            {rows.map((row) => (
              <div key={row.i} className="grid gap-1 px-6 py-4 sm:grid-cols-2 sm:gap-6">
                <p className="whitespace-pre-wrap text-sm text-muted-foreground">{row.src}</p>
                <p className="whitespace-pre-wrap text-sm">
                  {row.error ? (
                    <span className="text-destructive">{row.error}</span>
                  ) : (
                    row.mt || "…"
                  )}
                </p>
              </div>
            ))}
          </CardContent>
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
