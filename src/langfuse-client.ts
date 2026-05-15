import { createHash } from "node:crypto";
import { LangfuseClient as LangfuseApiClient } from "@langfuse/client";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import {
	type LangfuseGeneration as SdkLangfuseGeneration,
	type LangfuseSpan as SdkLangfuseSpan,
	setLangfuseTracerProvider,
	startObservation,
} from "@langfuse/tracing";
import type { SpanContext } from "@opentelemetry/api";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import type { Config } from "./config.js";

type LangfuseMetadata = Record<string, unknown>;

type TraceUpdateBody = {
	id?: string | null;
	name?: string;
	metadata?: LangfuseMetadata;
	output?: unknown;
	input?: unknown;
	sessionId?: string;
	userId?: string;
	tags?: string[];
	release?: string;
	version?: string;
	environment?: string;
	public?: boolean;
};

type ObservationEndBody = {
	metadata?: LangfuseMetadata;
	isError?: boolean;
	output?: unknown;
	usage?: unknown;
	usageDetails?: Record<string, number>;
	costDetails?: Record<string, number>;
	statusMessage?: string;
};

type ObservationUpdateBody = {
	metadata?: LangfuseMetadata;
	input?: unknown;
	output?: unknown;
	usage?: unknown;
	usageDetails?: Record<string, number>;
	costDetails?: Record<string, number>;
	statusMessage?: string;
};

export interface LangfuseTrace {
	id: string;
	update(body?: TraceUpdateBody): void;
}

export interface LangfuseSpan {
	id: string;
	update?(body: ObservationUpdateBody): void;
	end(body?: ObservationEndBody): void;
}

export interface LangfuseGeneration {
	id: string;
	update?(body: ObservationUpdateBody): void;
	end(body?: ObservationEndBody): void;
}

export interface LangfuseClient {
	trace(body?: {
		id?: string | null;
		name: string;
		metadata?: LangfuseMetadata;
		input?: unknown;
		output?: unknown;
		sessionId?: string;
		userId?: string;
		tags?: string[];
		release?: string;
		version?: string;
		environment?: string;
		public?: boolean;
	}): LangfuseTrace;
	span(body: {
		name: string;
		traceId: string;
		parentObservationId?: string;
		metadata?: LangfuseMetadata;
		input?: unknown;
		output?: unknown;
	}): LangfuseSpan;
	generation(body: {
		name: string;
		traceId: string;
		parentObservationId?: string;
		metadata?: LangfuseMetadata;
		input?: unknown;
		output?: unknown;
		usage?: unknown;
		usageDetails?: Record<string, number>;
		model?: string;
		costDetails?: Record<string, number>;
		version?: string;
	}): LangfuseGeneration;
	score(body: {
		name: string;
		value: number;
		traceId?: string;
		observationId?: string;
		sessionId?: string;
		comment?: string;
	}): void;
	flushAsync?(): Promise<void>;
	shutdownAsync(): Promise<void>;
}

type SdkObservation = SdkLangfuseSpan | SdkLangfuseGeneration;

class LangfuseV4Client implements LangfuseClient {
	private readonly apiClient: LangfuseApiClient;
	private readonly spanProcessor: LangfuseSpanProcessor;
	private readonly tracerProvider: BasicTracerProvider;
	private readonly observationsById = new Map<string, SdkObservation>();
	private readonly rootsByTraceId = new Map<string, SdkLangfuseSpan>();
	private activeOperations = 0;
	private idleWaiters: Array<() => void> = [];
	private shutdownPromise: Promise<void> | null = null;
	private shuttingDown = false;

	get isShuttingDown() {
		return this.shuttingDown;
	}

	constructor(config: Config) {
		this.apiClient = new LangfuseApiClient({
			publicKey: config.publicKey,
			secretKey: config.secretKey,
			baseUrl: config.host,
		});
		this.spanProcessor = new LangfuseSpanProcessor({
			publicKey: config.publicKey,
			secretKey: config.secretKey,
			baseUrl: config.host,
			environment: config.environment || undefined,
			release: config.release || undefined,
		});
		this.tracerProvider = new BasicTracerProvider({
			spanProcessors: [this.spanProcessor],
		});
		setLangfuseTracerProvider(this.tracerProvider);
	}

	trace(body?: TraceUpdateBody): LangfuseTrace {
		const finishOperation = this.beginOperation();
		try {
			const traceId = normalizeTraceId(body?.id);
			const root = startObservation(
				body?.name ?? "trace",
				toObservationAttributes(body),
				{
					asType: "span",
					parentSpanContext: traceId
						? createParentSpanContext(traceId)
						: undefined,
				},
			);
			root.updateTrace(toTraceAttributes(body));
			this.remember(root);
			this.rootsByTraceId.set(root.traceId, root);

			return {
				id: root.traceId,
				update: (updateBody?: TraceUpdateBody) => {
					root.update(toObservationAttributes(updateBody));
					root.updateTrace(toTraceAttributes(updateBody));
				},
			};
		} finally {
			finishOperation();
		}
	}

	span(body: {
		name: string;
		traceId: string;
		parentObservationId?: string;
		metadata?: LangfuseMetadata;
		input?: unknown;
		output?: unknown;
	}): LangfuseSpan {
		const finishOperation = this.beginOperation();
		try {
			const observation = this.startChildObservation(body, "span");
			return this.wrapObservation(observation);
		} finally {
			finishOperation();
		}
	}

	generation(body: {
		name: string;
		traceId: string;
		parentObservationId?: string;
		metadata?: LangfuseMetadata;
		input?: unknown;
		output?: unknown;
		usage?: unknown;
		usageDetails?: Record<string, number>;
		model?: string;
		costDetails?: Record<string, number>;
		version?: string;
	}): LangfuseGeneration {
		const finishOperation = this.beginOperation();
		try {
			const observation = this.startChildObservation(body, "generation");
			return this.wrapObservation(observation);
		} finally {
			finishOperation();
		}
	}

	score(body: {
		name: string;
		value: number;
		traceId?: string;
		observationId?: string;
		sessionId?: string;
		comment?: string;
	}) {
		this.apiClient.score.create(body);
	}

	async flushAsync() {
		await this.tracerProvider.forceFlush();
		await this.apiClient.flush();
	}

	async shutdownAsync() {
		if (this.shutdownPromise) return this.shutdownPromise;
		this.shuttingDown = true;
		this.shutdownPromise = this.shutdown();
		return this.shutdownPromise;
	}

	private async shutdown() {
		await this.waitForIdle();
		await this.flushAsync();
		await this.tracerProvider.shutdown();
		await this.apiClient.shutdown();
		this.observationsById.clear();
		this.rootsByTraceId.clear();
	}

	private startChildObservation(
		body: ObservationUpdateBody & {
			name: string;
			traceId: string;
			parentObservationId?: string;
			model?: string;
			version?: string;
		},
		asType: "span" | "generation",
	) {
		this.assertNotShuttingDown();
		const parent =
			(body.parentObservationId
				? this.observationsById.get(body.parentObservationId)
				: undefined) ?? this.rootsByTraceId.get(body.traceId);

		const attributes = toObservationAttributes(body);
		const observation =
			asType === "generation"
				? parent
					? parent.startObservation(body.name, attributes, {
							asType: "generation",
						})
					: startObservation(body.name, attributes, {
							asType: "generation",
							parentSpanContext: createParentSpanContext(body.traceId),
						})
				: parent
					? parent.startObservation(body.name, attributes)
					: startObservation(body.name, attributes, {
							parentSpanContext: createParentSpanContext(body.traceId),
						});
		this.remember(observation);
		return observation;
	}

	private wrapObservation<T extends SdkObservation>(observation: T) {
		return {
			id: observation.id,
			update: (body: ObservationUpdateBody) => {
				observation.update(toObservationAttributes(body));
			},
			end: (body?: ObservationEndBody) => {
				observation.update(toObservationAttributes(body));
				observation.end();
			},
		};
	}

	private remember(observation: SdkObservation) {
		this.observationsById.set(observation.id, observation);
	}

	private beginOperation() {
		this.assertNotShuttingDown();
		this.activeOperations += 1;
		let finished = false;
		return () => {
			if (finished) return;
			finished = true;
			this.activeOperations -= 1;
			if (this.activeOperations === 0) {
				for (const resolve of this.idleWaiters) resolve();
				this.idleWaiters = [];
			}
		};
	}

	private waitForIdle() {
		if (this.activeOperations === 0) return Promise.resolve();
		return new Promise<void>((resolve) => {
			this.idleWaiters.push(resolve);
		});
	}

	private assertNotShuttingDown() {
		if (this.shuttingDown) {
			throw new Error("Langfuse client is shutting down");
		}
	}
}

let client: LangfuseClient | null = null;
let clientConfigKey = "";

export async function flushClient() {
	if (client?.flushAsync) {
		await client.flushAsync();
	}
}

export async function shutdownClient() {
	if (client) {
		const currentClient = client;
		await currentClient.shutdownAsync();
		if (client === currentClient) {
			client = null;
			clientConfigKey = "";
		}
	}
}

export async function getClient(config: Config): Promise<LangfuseClient> {
	const nextConfigKey = JSON.stringify({
		publicKey: config.publicKey,
		secretKey: config.secretKey,
		host: config.host,
		release: config.release,
		environment: config.environment,
	});

	if (client && clientConfigKey !== nextConfigKey) {
		await shutdownClient();
	}

	if (client instanceof LangfuseV4Client && client.isShuttingDown) {
		await shutdownClient();
	}

	if (!client) {
		client = new LangfuseV4Client(config);
		clientConfigKey = nextConfigKey;
	}

	return client;
}

function toObservationAttributes(
	body?: ObservationUpdateBody & {
		isError?: boolean;
		model?: string;
		version?: string;
		environment?: string;
	},
) {
	return {
		input: body?.input,
		output: body?.output,
		metadata: body?.metadata,
		level: body?.isError ? ("ERROR" as const) : undefined,
		statusMessage: body?.statusMessage,
		usageDetails: body?.usageDetails ?? normalizeUsageDetails(body?.usage),
		costDetails: body?.costDetails,
		model: body?.model,
		version: body?.version,
		environment: body?.environment,
	};
}

function toTraceAttributes(body?: TraceUpdateBody) {
	return {
		name: body?.name,
		userId: body?.userId,
		sessionId: body?.sessionId,
		version: body?.version,
		release: body?.release,
		input: body?.input,
		output: body?.output,
		metadata: body?.metadata,
		tags: body?.tags,
		public: body?.public,
		environment: body?.environment,
	};
}

function normalizeUsageDetails(usage: unknown) {
	if (!usage || typeof usage !== "object") return undefined;
	const usageRecord = usage as Record<string, unknown>;
	const details: Record<string, number> = {};
	for (const key of ["input", "output", "total"]) {
		const value = usageRecord[key];
		if (typeof value === "number") details[key] = value;
	}
	return Object.keys(details).length > 0 ? details : undefined;
}

function normalizeTraceId(id?: string | null) {
	if (!id) return undefined;
	const hexId = id.replaceAll("-", "").toLowerCase();
	if (/^[0-9a-f]{32}$/.test(hexId) && !/^0+$/.test(hexId)) {
		return hexId;
	}
	const hashed = createHash("sha256").update(id).digest("hex").slice(0, 32);
	return /^0+$/.test(hashed) ? "00000000000000000000000000000001" : hashed;
}

function createParentSpanContext(traceId: string): SpanContext {
	return {
		traceId: normalizeTraceId(traceId) ?? traceId,
		spanId: "0000000000000001",
		traceFlags: 1,
	};
}
