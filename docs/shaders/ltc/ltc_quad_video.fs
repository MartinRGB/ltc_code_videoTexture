// bind roughness   {label:"Roughness", default:0.5, min:0.01, max:5, step:0.001}
// bind dcolor      {label:"Diffuse Color",  r:1.0, g:1.0, b:1.0}
// bind scolor      {label:"Specular Color", r:1.0, g:1.0, b:1.0}
// bind intensity   {label:"Light Intensity", default:4, min:0, max:100}
// bind width       {label:"Width",  default: 14.4, min:0.1, max:15, step:0.1}
// bind height      {label:"Height", default: 9, min:0.1, max:15, step:0.1}
// bind roty        {label:"Rotation Y", default: 0, min:0, max:1, step:0.001}
// bind rotz        {label:"Rotation Z", default: 0, min:0, max:1, step:0.001}
// bind twoSided    {label:"Two-sided", default:false}
// bind clipless    {label:"Clipless Approximation", default:false}
// bind isVideo     {label:"Is Video", default:true}

uniform float roughness;
uniform vec3  dcolor;
uniform vec3  scolor;

uniform float intensity;
uniform float width;
uniform float height;
uniform float roty;
uniform float rotz;

uniform bool twoSided;
uniform bool clipless;

uniform sampler2D ltc_1;
uniform sampler2D ltc_2;
uniform sampler2D img_tex;
uniform sampler2D floor_tex;
uniform sampler2D vid_tex;

uniform mat4  view;
uniform vec2  resolution;
uniform int   sampleCount;

const float LUT_SIZE  = 64.0;
const float LUT_SCALE = (LUT_SIZE - 1.0)/LUT_SIZE;
const float LUT_BIAS  = 0.5/LUT_SIZE;

const float pi = 3.14159265;

// Tracing and intersection
///////////////////////////

struct Ray
{
    vec3 origin;
    vec3 dir;
};

struct Rect
{
    vec3  center;
    vec3  dirx;
    vec3  diry;
    float halfx;
    float halfy;

    vec4  plane;
};

bool RayPlaneIntersect(Ray ray, vec4 plane, out float t)
{
    t = -dot(plane, vec4(ray.origin, 1.0))/dot(plane.xyz, ray.dir);
    return t > 0.0;
}

bool RayRectIntersect(Ray ray, Rect rect, out float t)
{
    bool intersect = RayPlaneIntersect(ray, rect.plane, t);
    if (intersect)
    {
        vec3 pos  = ray.origin + ray.dir*t;
        vec3 lpos = pos - rect.center;

        float x = dot(lpos, rect.dirx);
        float y = dot(lpos, rect.diry);

        if (abs(x) > rect.halfx || abs(y) > rect.halfy)
            intersect = false;
    }

    return intersect;
}

vec4 RayRectIntersectDrawColor(Ray ray, Rect rect, out float t)
{
    bool intersect = RayPlaneIntersect(ray, rect.plane, t);
    if (intersect)
    {
        vec3 pos  = ray.origin + ray.dir*t;
        vec3 lpos = pos - rect.center;

        float x = dot(lpos, rect.dirx);
        float y = dot(lpos, rect.diry);

        if (abs(x) > rect.halfx || abs(y) > rect.halfy){
            return vec4(0.);
        }
        else{
            float vX = (x/(rect.halfx) + 1.)/2.;
            float vY = (y/(rect.halfy) + 1.)/2.;
            vec2 vUv = vec2(vX,vY);
            vec4 col = texture(vid_tex,vec2(vUv.x,1.-vUv.y));
            return col;
        }
    }

}

// Camera functions
///////////////////

Ray GenerateCameraRay()
{
    Ray ray;

    vec2 xy = 2.0*gl_FragCoord.xy/resolution - vec2(1.0);

    ray.dir = normalize(vec3(xy, 2.0));

    float focalDistance = 2.0;
    float ft = focalDistance/ray.dir.z;
    vec3 pFocus = ray.dir*ft;

    ray.origin = vec3(0);
    ray.dir    = normalize(pFocus - ray.origin);

    // Apply camera transform
    ray.origin = (view*vec4(ray.origin, 1)).xyz;
    ray.dir    = (view*vec4(ray.dir,    0)).xyz;

    return ray;
}

vec3 mul(mat3 m, vec3 v)
{
    return m * v;
}

mat3 mul(mat3 m1, mat3 m2)
{
    return m1 * m2;
}

vec3 rotation_y(vec3 v, float a)
{
    vec3 r;
    r.x =  v.x*cos(a) + v.z*sin(a);
    r.y =  v.y;
    r.z = -v.x*sin(a) + v.z*cos(a);
    return r;
}

vec3 rotation_z(vec3 v, float a)
{
    vec3 r;
    r.x =  v.x*cos(a) - v.y*sin(a);
    r.y =  v.x*sin(a) + v.y*cos(a);
    r.z =  v.z;
    return r;
}

vec3 rotation_yz(vec3 v, float ay, float az)
{
    return rotation_z(rotation_y(v, ay), az);
}

// ******* Filtered Border Region 
// https://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf p-104  -> filtered border region
// https://www.shadertoy.com/view/dd2SDd

float maskBox(vec2 _st, vec2 _size, float _smoothEdges){
    _size = vec2(0.5)-_size*0.5;
    vec2 aa = vec2(_smoothEdges*0.5);
    vec2 uv = smoothstep(_size,_size+aa,_st);
    uv *= smoothstep(_size,_size+aa,vec2(1.0)-_st);
    return uv.x*uv.y;
}

vec4 draw(vec2 uv,in sampler2D tex) {
    // return texture(tex,vec2(1.- uv.x,uv.y)).rgb;   
    return textureLod(tex,vec2(uv.x,1. - uv.y),8.);   
}

float grid(float var, float size) {
    return floor(var*size)/size;
}

float blurRand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// *** the 'repeats' affect the performance
vec4 blurredImage( in float roughness,in vec2 uv , in sampler2D tex)
{
    
    float bluramount = 0.2 * roughness;
    //float dists = 5.;
    vec4 blurred_image = vec4(0.);
    #define repeats 30.
    for (float i = 0.; i < repeats; i++) { 
        //Older:
        //vec2 q = vec2(cos(degrees((grid(i,dists)/repeats)*360.)),sin(degrees((grid(i,dists)/repeats)*360.))) * (1./(1.+mod(i,dists)));
        vec2 q = vec2(cos(degrees((i/repeats)*360.)),sin(degrees((i/repeats)*360.))) *  (blurRand(vec2(i,uv.x+uv.y))+bluramount); 
        vec2 uv2 = uv+(q*bluramount);
        blurred_image += draw(uv2,tex)/2.;
        //One more to hide the noise.
        q = vec2(cos(degrees((i/repeats)*360.)),sin(degrees((i/repeats)*360.))) *  (blurRand(vec2(i+2.,uv.x+uv.y+24.))+bluramount); 
        uv2 = uv+(q*bluramount);
        blurred_image += draw(uv2,tex)/2.;
    }
    blurred_image /= repeats;
        
    return blurred_image;
}


vec4 filterBorderRegion(in float roughness,in vec2 uv,in sampler2D tex){
    // this is useless now
	float scale = 1.;
    float error = 0.4; //0.45

    // Convert uv range to -1 to 1
    vec2 UVC = uv * 2.0 - 1.0;
    UVC *= (1. * 0.5 + 0.5) * (1. + (1. - scale));
    // Convert back to 0 to 1 range
    UVC = UVC * 0.5 + 0.5;

    vec4 ClearCol;
    vec4 BlurCol;
    
    BlurCol = blurredImage(2.,uv,tex);
	if(UVC.x < 1. && UVC.x > 0. && UVC.y > 0. && UVC.y < 1.){
        ClearCol = blurredImage(min(2.,roughness),UVC,tex);
    }
	//ClearCol.rgb = blurredImage(roughness,UVC,tex);
	float boxMask = maskBox(UVC,vec2(scale+0.),error);
    BlurCol.rgb = mix(BlurCol.rgb, ClearCol.rgb, boxMask);
    return BlurCol;
    
    // # Method 2
	//return blurredImage(min(2.,roughness),uv,tex).rgb;
}

vec4 FetchDiffuseFilteredTexture(float roughness,vec3 L[5],vec3 vLooupVector,sampler2D tex)
{
	vec3 V1 = L[1] - L[0];
	vec3 V2 = L[3] - L[0];
	// Plane's normal
	vec3 PlaneOrtho = cross(V1, V2);
	float PlaneAreaSquared = dot(PlaneOrtho, PlaneOrtho);
	float planeDistxPlaneArea = dot(PlaneOrtho, L[0]);
	// orthonormal projection of (0,0,0) in area light space
	vec3 P = planeDistxPlaneArea * PlaneOrtho / PlaneAreaSquared - L[0];

	// find tex coords of P
	float dot_V1_V2 = dot(V1, V2);
	float inv_dot_V1_V1 = 1.0 / dot(V1, V1);
	vec3 V2_ = V2 - V1 * dot_V1_V2 * inv_dot_V1_V1;
	vec2 UV;
	UV.y = dot(V2_, P) / dot(V2_, V2_);
	UV.x = dot(V1, P) * inv_dot_V1_V1 - dot_V1_V2 * inv_dot_V1_V1 * UV.y;

	// float scale = 1.;
    // float error = 0.45;
    // // Convert uv range to -1 to 1
    // vec2 UVC = UV * 2.0 - 1.0;
    // UVC *= (1. * 0.5 + 0.5) * (1. + (1. - scale));
    // // Convert back to 0 to 1 range
    // UVC = UVC * 0.5 + 0.5;

    // vec4 ClearCol;
    // vec4 BlurCol;
    
    // BlurCol.rgb = blurredImage(2.,UV,tex);
	// if(UVC.x < 1. && UVC.x > 0. && UVC.y > 0. && UVC.y < 1.){
    //     ClearCol.rgb = blurredImage(min(2.,roughness),UVC,tex);
    // }
	// //ClearCol.rgb = blurredImage(roughness,UVC,tex);
	// float boxMask = maskBox(UVC,vec2(scale+0.),error);
    // BlurCol.rgb = mix(BlurCol.rgb, ClearCol.rgb, boxMask);

    // to delete border light even the canvas is dark
    // UV -= .5;
    // UV /= 1.1;
    // UV += .5;

	return filterBorderRegion(roughness,UV,tex);
}

// Linearly Transformed Cosines
///////////////////////////////

vec3 IntegrateEdgeVec(vec3 v1, vec3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206*y)*y;
    float b = 3.4175940 + (4.1616724 + y)*y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5*inversesqrt(max(1.0 - x*x, 1e-7)) - v;

    return cross(v1, v2)*theta_sintheta;
}

float IntegrateEdge(vec3 v1, vec3 v2)
{
    return IntegrateEdgeVec(v1, v2).z;
}

void ClipQuadToHorizon(inout vec3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }

    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}


mat3 caculatedMInv(float roughness,vec3 N,vec3 V,in sampler2D lut_tex){

    const float PI = 3.1415926;
    const float LUTSIZE  = 64.0;
    const float MATRIX_PARAM_OFFSET = 64.0;

    float theta = acos(dot(N, V));
    
    vec2 uv = vec2(roughness, theta/(0.5*PI)) * float(LUTSIZE-1.);
    uv += vec2(0.5 );
    
    vec2 LUT_RES = vec2(64.);
    vec4 params = texture(lut_tex, (uv+vec2(MATRIX_PARAM_OFFSET, 0.0))/LUT_RES);
    
    mat3 Minv = mat3(
        vec3(  1,        0,      params.y),
        vec3(  0,     params.z,   0),
        vec3(params.w,   0,      params.x)
    );
    
    return Minv;
}


vec3 LTC_Evaluate(
    vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided)
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    Minv = mul(Minv, transpose(mat3(T1, T2, N)));

    // polygon (allocate 5 vertices for clipping)
    vec3 L[5];
    L[0] = mul(Minv, points[0] - P);
    L[1] = mul(Minv, points[1] - P);
    L[2] = mul(Minv, points[2] - P);
    L[3] = mul(Minv, points[3] - P);

    // integrate
    float sum = 0.0;

    if (clipless)
    {
        vec3 dir = points[0].xyz - P;
        vec3 lightNormal = cross(points[1] - points[0], points[3] - points[0]);
        bool behind = (dot(dir, lightNormal) < 0.0);

        L[0] = normalize(L[0]);
        L[1] = normalize(L[1]);
        L[2] = normalize(L[2]);
        L[3] = normalize(L[3]);

        vec3 vsum = vec3(0.0);

        vsum += IntegrateEdgeVec(L[0], L[1]);
        vsum += IntegrateEdgeVec(L[1], L[2]);
        vsum += IntegrateEdgeVec(L[2], L[3]);
        vsum += IntegrateEdgeVec(L[3], L[0]);

        float len = length(vsum);
        float z = vsum.z/len;

        if (behind)
            z = -z;

        vec2 uv = vec2(z*0.5 + 0.5, len);
        uv = uv*LUT_SCALE + LUT_BIAS;

        float scale = texture(ltc_2, uv).w;

        sum = len*scale;

        if (behind && !twoSided)
            sum = 0.0;
    }
    else
    {
        int n;
        ClipQuadToHorizon(L, n);

        if (n == 0)
            return vec3(0, 0, 0);
        // project onto sphere
        L[0] = normalize(L[0]);
        L[1] = normalize(L[1]);
        L[2] = normalize(L[2]);
        L[3] = normalize(L[3]);
        L[4] = normalize(L[4]);

        // integrate
        sum += IntegrateEdge(L[0], L[1]);
        sum += IntegrateEdge(L[1], L[2]);
        sum += IntegrateEdge(L[2], L[3]);
        if (n >= 4)
            sum += IntegrateEdge(L[3], L[4]);
        if (n == 5)
            sum += IntegrateEdge(L[4], L[0]);

        sum = twoSided ? abs(sum) : max(0.0, sum);
    }

    vec3 Lo_i = vec3(sum, sum, sum);
    
    vec3 PL[5];
    PL[0] = mul(Minv, points[0] - P);
    PL[1] = mul(Minv, points[1] - P);
    PL[2] = mul(Minv, points[2] - P);
    PL[3] = mul(Minv, points[3] - P);

    // *** insert code here ***
    vec3 e1 = normalize(PL[0] - PL[1]);
    vec3 e2 = normalize(PL[2] - PL[1]);
    vec3 N2 = cross(e1, e2); // Normal to light
    vec3 V2 = N2 * dot(PL[1], N2); // Vector to some point in light rect
    vec2 Tlight_shape = vec2(length(PL[0] - PL[1]), length(PL[2] - PL[1]));
    V2 = V2 - PL[1];
    float b = e1.y*e2.x - e1.x*e2.y + .1; // + .1 to remove artifacts
	vec2 pLight = vec2((V2.y*e2.x - V2.x*e2.y)/b, (V2.x*e1.y - V2.y*e1.x)/b);
   	pLight /= Tlight_shape;
    //vec4 texCol = texture(img_tex, vec2(pLight.x,1.-pLight.y));
    vec4 ref_col = FetchDiffuseFilteredTexture(roughness,PL,vec3(sum),vid_tex);
    return Lo_i*ref_col.rgb;
}

// Scene helpers
////////////////

void InitRect(out Rect rect)
{
    rect.dirx = rotation_yz(vec3(1, 0, 0), roty*2.0*pi, rotz*2.0*pi);
    rect.diry = rotation_yz(vec3(0, 1, 0), roty*2.0*pi, rotz*2.0*pi);

    rect.center = vec3(0, 6, 32);
    rect.halfx  = 0.5*width;
    rect.halfy  = 0.5*height;

    vec3 rectNormal = cross(rect.dirx, rect.diry);
    rect.plane = vec4(rectNormal, -dot(rectNormal, rect.center));
}

void InitRectPoints(Rect rect, out vec3 points[4])
{
    vec3 ex = rect.halfx*rect.dirx;
    vec3 ey = rect.halfy*rect.diry;

    points[0] = rect.center - ex - ey;
    points[1] = rect.center + ex - ey;
    points[2] = rect.center + ex + ey;
    points[3] = rect.center - ex + ey;
}

// Misc. helpers
////////////////

float saturate(float v)
{
    return clamp(v, 0.0, 1.0);
}

vec3 PowVec3(vec3 v, float p)
{
    return vec3(pow(v.x, p), pow(v.y, p), pow(v.z, p));
}

const float gamma = 2.2;
vec3 ToLinear(vec3 v) { return PowVec3(v, gamma); }

out vec4 FragColor;

void main()
{
    Rect rect;
    InitRect(rect);

    vec3 points[4];
    InitRectPoints(rect, points);

    vec4 floorPlane = vec4(0, 1, 0, 0);

    vec3 lcol = vec3(intensity);
    vec3 dcol = ToLinear(dcolor);
    vec3 scol = ToLinear(scolor);

    vec3 col = vec3(0);

    Ray ray = GenerateCameraRay();

    float distToFloor;
    bool hitFloor = RayPlaneIntersect(ray, floorPlane, distToFloor);
    if (hitFloor)
    {
        vec3 pos = ray.origin + ray.dir*distToFloor;

        vec3 N = floorPlane.xyz;
        vec3 V = -ray.dir;


        float new_Roughness;
        vec3 floorTexture = texture(floor_tex,pos.xz/10.).rgb;

        new_Roughness = floorTexture.x;
        new_Roughness *= roughness;
        new_Roughness += 0.2;
        

        float ndotv = saturate(dot(N, V));
        vec2 uv = vec2(new_Roughness, sqrt(1.0 - ndotv)); //roughness
        uv = uv*LUT_SCALE + LUT_BIAS;

        vec4 t1 = texture(ltc_1, uv);
        vec4 t2 = texture(ltc_2, uv);

        mat3 Minv = mat3(
            vec3(t1.x, 0, t1.y),
            vec3(  0,  1,    0),
            vec3(t1.z, 0, t1.w)
        );

        //Minv = caculatedMInv(new_Roughness,N,V,ltc_1);


        vec3 spec = LTC_Evaluate(N, V, pos, Minv, points, twoSided);
        // BRDF shadowing and Fresnel
        spec *= scol*t2.x + (1.0 - scol)*t2.y;

        vec3 diff = LTC_Evaluate(N, V, pos, mat3(1), points, twoSided);

        // col - lcol*(spec + dcol*diff);
        col = lcol*(spec + dcol*diff) * floorTexture;
        col *= 1.0;
    }

    float distToRect;
    if (RayRectIntersect(ray, rect, distToRect))
        if ((distToRect < distToFloor) || !hitFloor)
            col = RayRectIntersectDrawColor(ray, rect, distToRect).rgb;

    FragColor = vec4(col, 1.0);
}