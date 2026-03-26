export interface AutomationConfig {
  enabled: boolean;
  eventUrl: string | null;
  runId: string | null;
  cameraLabelContains: string | null;
  autoStartVideo: boolean;
  autoLeaveSeconds: number | null;
}

export function getAutomationConfig(searchParams: URLSearchParams): AutomationConfig {
  const autoLeaveRaw = searchParams.get('autoLeaveSeconds');
  const autoLeaveSeconds = autoLeaveRaw ? Number(autoLeaveRaw) : NaN;
  const eventUrl = searchParams.get('automationEventUrl')?.trim() || null;
  const cameraLabelContains = searchParams.get('cameraLabelContains')?.trim() || null;
  const autoStartVideo = searchParams.get('autoStartVideo') === '1';
  const autoLeave =
    Number.isFinite(autoLeaveSeconds) && autoLeaveSeconds > 0 ? autoLeaveSeconds : null;

  return {
    enabled: Boolean(eventUrl || cameraLabelContains || autoStartVideo || autoLeave),
    eventUrl,
    runId: searchParams.get('runId'),
    cameraLabelContains,
    autoStartVideo,
    autoLeaveSeconds: autoLeave
  };
}

export async function logAutomationEvent(
  config: AutomationConfig,
  event: string,
  detail: Record<string, unknown> = {}
) {
  const payload = {
    event,
    runId: config.runId,
    detail,
    clientTimeIso: new Date().toISOString(),
    location: `${window.location.pathname}${window.location.search}`
  };

  console.log(`[automation] ${event}`, payload);

  if (!config.eventUrl) {
    return;
  }

  try {
    await fetch(config.eventUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      keepalive: true
    });
  } catch (error) {
    console.warn('[automation] event post failed', event, error);
  }
}

export function findDeviceByLabel<T extends { deviceId: string; label: string }>(
  devices: T[],
  labelContains: string | null
): T | undefined {
  if (!labelContains) {
    return undefined;
  }
  const needle = labelContains.toLowerCase();
  return devices.find((device) => device.label.toLowerCase().includes(needle));
}
